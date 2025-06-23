# DiT with cross attention

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
from flash_attn_interface import flash_attn_varlen_func


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


def prepare_flash_attention_inputs(hidden_states, attention_mask=None):
    """
    Prepares inputs for flash_attn_varlen_func by flattening and calculating cu_seqlens.

    Args:
        hidden_states: Tensor of shape (batch_size, max_seqlen, embed_dim)
        attention_mask: Tensor of shape (batch_size, max_seqlen) (0 for padding, 1 for actual token)

    Returns:
        flattened_hidden_states: Tensor of shape (total_num_tokens, embed_dim)
        cu_seqlens: Tensor of shape (batch_size + 1,)
        max_seqlen_in_batch: Max sequence length in the current batch
        indices: Indices to re-construct the original padded tensor (for unflattening)
    """
    batch_size, max_seqlen_in_batch, embed_dim = hidden_states.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, max_seqlen_in_batch), device=hidden_states.device)

    # Calculate actual sequence lengths from the attention mask
    sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)

    # Create cumulative sequence lengths
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
        sequence_lengths.cumsum(dim=0, dtype=torch.int32)
    ])

    # Flatten hidden_states to (total_num_tokens, embed_dim)
    # This removes padding tokens
    # Using torch.nonzero and index_select is correct for this.
    indices = torch.nonzero(attention_mask.view(-1), as_tuple=True)[0]
    flattened_hidden_states = torch.index_select(hidden_states.view(-1, embed_dim), 0, indices)

    return flattened_hidden_states, cu_seqlens, max_seqlen_in_batch, indices


def unprepare_flash_attention_outputs(output_flat, indices, batch_size, max_seqlen, hidden_size):
    """
    Reconstructs the padded output tensor from the flattened output.

    Args:
        output_flat: Tensor of shape (total_num_tokens, embed_dim)
        indices: Indices used during flattening to reconstruct the original shape
        batch_size: Original batch size
        max_seqlen: Original maximum sequence length
        embed_dim: Original embedding dimension
        dtype: Desired output data type
        device: Desired output device
    Returns:
        padded_output: Tensor of shape (batch_size, max_seqlen, embed_dim)
    """
    output = torch.zeros(
        (batch_size * max_seqlen, hidden_size),
        dtype=output_flat.dtype,
        device=output_flat.device
    )
    output.index_copy_(0, indices, output_flat) # output_flat should already be (total_tokens, embed_dim)
    output = output.view(batch_size, max_seqlen, hidden_size)
    return output


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, trainable=False):
        super().__init__()
        self.eps = eps
        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            return (x * norm * self.weight).to(dtype=x_dtype)
        else:
            return (x * norm).to(dtype=x_dtype)
    
    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)


class QKNorm(nn.Module):
    """Normalizing the query and the key independently, as Flux proposes"""

    def __init__(self, dim, trainable=False):
        super().__init__()
        self.query_norm = RMSNorm(dim, trainable=trainable)
        self.key_norm = RMSNorm(dim, trainable=trainable)

    def forward(self, q, k):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

    def reset_parameters(self):
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        is_self_attn=True,
        dynamic_softmax_temperature=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_self_attn = is_self_attn
        self.dynamic_softmax_temperature = dynamic_softmax_temperature

        if is_self_attn:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.context_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=False)

        self.qk_norm = QKNorm(self.head_dim)

    def forward(self, x, x_cu_seqlens, x_max_seqlen_in_batch, context=None, context_cu_seqlens=None, context_max_seqlen_in_batch=None, rope=None):
        if self.is_self_attn:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "l (k h d) -> k h l d", k=3, h=self.num_heads)
            q, k, v = qkv.unbind(0)

            if rope is not None:
                # print(q.shape, rope[0].shape, rope[1].shape)
                q = apply_rotary_emb(q, rope[0], rope[1])
                k = apply_rotary_emb(k, rope[0], rope[1])

                # https://arxiv.org/abs/2306.08645
                # https://arxiv.org/abs/2410.01104
                # ratioonale is that if tokens get larger, categorical distribution get more uniform
                # so you want to enlargen entropy.

                token_length = q.shape[2]
                if self.dynamic_softmax_temperature:
                    ratio = math.sqrt(math.log(token_length) / math.log(1040.0))  # 1024 + 16
                    k = k * ratio
            q, k = self.qk_norm(q, k)
            q = rearrange(q, "h l d -> l h d")
            k = rearrange(k, "h l d -> l h d")
            v = rearrange(v, "h l d -> l h d")
            cu_seqlens_q = x_cu_seqlens
            cu_seqlens_k = x_cu_seqlens
            max_seqlen_q = x_max_seqlen_in_batch
            max_seqlen_k = x_max_seqlen_in_batch
        else:
            q = rearrange(self.q(x), "l (h d) -> l h d", h=self.num_heads)
            kv = rearrange(
                self.context_kv(context),
                "l (k h d) -> k l h d",
                k=2,
                h=self.num_heads,
            )
            k, v = kv.unbind(0)
            q, k = self.qk_norm(q, k)
            cu_seqlens_q = x_cu_seqlens
            cu_seqlens_k = context_cu_seqlens
            max_seqlen_q = x_max_seqlen_in_batch
            max_seqlen_k = context_max_seqlen_in_batch

        x, _ = flash_attn_varlen_func(
            q, k, v, 
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale
        )
        x = rearrange(x, "l h d -> l (h d)")
        x = self.proj(x)
        return x

    def reset_parameters(self):
        if hasattr(self, 'qkv'):
            self.qkv.reset_parameters()
        if hasattr(self, 'q'):
            self.q.reset_parameters()
        if hasattr(self, 'context_kv'):
            self.context_kv.reset_parameters()
        self.proj.reset_parameters()
        self.qk_norm.reset_parameters()


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        do_cross_attn=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        dynamic_softmax_temperature=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = LigerRMSNorm(hidden_size)
        self.self_attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            is_self_attn=True,
            dynamic_softmax_temperature=dynamic_softmax_temperature,
        )

        if do_cross_attn:
            self.norm2 = LigerRMSNorm(hidden_size)
            self.cross_attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                is_self_attn=False,
                dynamic_softmax_temperature=dynamic_softmax_temperature,
            )
        else:
            self.norm2 = None
            self.cross_attn = None

        self.norm3 = LigerRMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        mlp_config = SimpleNamespace(
            hidden_size=hidden_size,
            intermediate_size=mlp_hidden,
            hidden_act="silu",
        )
        self.mlp = LigerSwiGLUMLP(mlp_config)

    # @torch.compile(mode='reduce-overhead')
    def forward(self, x, x_cu_seqlens, x_max_seqlen_in_batch, context, context_cu_seqlens, context_max_seqlen_in_batch, modulation, rope=None):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation

        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_sa) + shift_sa
        attn_out = self.self_attn(
            norm_x, x_cu_seqlens, x_max_seqlen_in_batch, 
            rope=rope
        )
        x = x + attn_out * gate_sa

        if self.norm2 is not None:
            norm_x = self.norm2(x)
            norm_x = norm_x * (1 + scale_ca) + shift_ca
            x = x + self.cross_attn(
                norm_x, x_cu_seqlens, x_max_seqlen_in_batch, 
                context, context_cu_seqlens, context_max_seqlen_in_batch
            ) * gate_ca

        norm_x = self.norm3(x)
        norm_x = norm_x * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(norm_x) * gate_mlp

        return x
    
    def reset_parameters(self):
        nn.init.ones_(self.norm1.weight)
        self.self_attn.reset_parameters()
        if self.norm2 is not None:
            nn.init.ones_(self.norm2.weight)
            self.cross_attn.reset_parameters()
        nn.init.ones_(self.norm3.weight)
        self.mlp.gate_proj.reset_parameters()
        self.mlp.up_proj.reset_parameters()
        self.mlp.down_proj.reset_parameters()



class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x
    
    def reset_parameters(self):
        self.patch_proj.reset_parameters()


class TwoDimRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, h=256, w=256):
        super().__init__()
        self.dim = dim
        self.base = base
        self.h = h
        self.w = w
        
        inv_freq = torch.FloatTensor([1.0 / (self.base ** (i / self.dim)) for i in range(0, self.dim, 2)])

        t_h = torch.arange(self.h, dtype=torch.float32)
        t_w = torch.arange(self.w, dtype=torch.float32)

        freqs_h = torch.outer(t_h, inv_freq).unsqueeze(1)  # h, 1, d / 2
        freqs_w = torch.outer(t_w, inv_freq).unsqueeze(0)  # 1, w, d / 2
        freqs_h = freqs_h.repeat(1, self.w, 1)  # h, w, d / 2
        freqs_w = freqs_w.repeat(self.h, 1, 1)  # h, w, d / 2
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)  # h, w, d

        self.register_buffer("freqs_hw_cos", freqs_hw.cos(), persistent=False)
        self.register_buffer("freqs_hw_sin", freqs_hw.sin(), persistent=False)

    def forward(self, x, height_width=None, extend_with_register_tokens=0):
        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        cos = self.freqs_hw_cos[0 : this_h, 0 : this_w]
        sin = self.freqs_hw_sin[0 : this_h, 0 : this_w]

        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)

        # append N of zero-attn tokens
        if extend_with_register_tokens > 0:
            cos = torch.cat(
                [
                    torch.ones(extend_with_register_tokens, cos.shape[1]).to(cos.device),
                    cos,
                ],
                0,
            )
            sin = torch.cat(
                [
                    torch.zeros(extend_with_register_tokens, sin.shape[1]).to(sin.device),
                    sin,
                ],
                0,
            )

        return cos[None, :, :], sin[None, :, :]  # [1, T + N, Attn-dim]

    def reset_parameters(self):
        inv_freq = torch.FloatTensor([1.0 / (self.base ** (i / self.dim)) for i in range(0, self.dim, 2)])

        t_h = torch.arange(self.h, dtype=torch.float32)
        t_w = torch.arange(self.w, dtype=torch.float32)

        freqs_h = torch.outer(t_h, inv_freq).unsqueeze(1)  # h, 1, d / 2
        freqs_w = torch.outer(t_w, inv_freq).unsqueeze(0)  # 1, w, d / 2
        freqs_h = freqs_h.repeat(1, self.w, 1)  # h, w, d / 2
        freqs_w = freqs_w.repeat(self.h, 1, 1)  # h, w, d / 2
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)  # h, w, d
        self.freqs_hw_cos[...] = freqs_hw.cos()
        self.freqs_hw_sin[...] = freqs_hw.sin()


def apply_rotary_emb(x, cos, sin):
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    assert x.ndim == 3  # multihead attention
    d = x.shape[2] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 2).to(dtype=orig_dtype)


class DiT(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):  # type: ignore[misc]
    @register_to_config
    def __init__(
        self,
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        train_bias_and_rms=True,
        use_rope=True,
        gradient_checkpoint=False,
        dynamic_softmax_temperature=False,
        rope_base=10000,
    ):
        super().__init__()

        self.context_proj = nn.Linear(cross_attn_input_size, hidden_size)
        self.context_norm = LigerRMSNorm(hidden_size)

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)

        if use_rope:
            self.rope = TwoDimRotary(hidden_size // (2 * num_heads), base=rope_base, h=512, w=512)
        else:
            self.positional_embedding = nn.Parameter(torch.zeros(1, 2048, hidden_size))

        self.register_tokens = nn.Parameter(torch.randn(1, 16, hidden_size))

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))
        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    do_cross_attn=(idx % 4 == 0 or idx < 8),     # cross attn every 4 blocks or first 8 blocks
                    qkv_bias=train_bias_and_rms,
                    dynamic_softmax_temperature=dynamic_softmax_temperature,
                )
                for idx in range(depth)
            ]
        )

        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        self.final_norm = RMSNorm(hidden_size, trainable=train_bias_and_rms)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        self.paramstatus = {}
        for n, p in self.named_parameters():
            self.paramstatus[n] = {
                "shape": p.shape,
                "requires_grad": p.requires_grad,
            }

    def save_lora_weights(self, save_directory):
        """Save LoRA weights to a file"""
        lora_state_dict = get_peft_model_state_dict(self)
        torch.save(lora_state_dict, f"{save_directory}/lora_weights.pt")

    def load_lora_weights(self, load_directory):
        """Load LoRA weights from a file"""
        lora_state_dict = torch.load(f"{load_directory}/lora_weights.pt")
        set_peft_model_state_dict(self, lora_state_dict)

    def reset_parameters(self):
        self.context_proj.reset_parameters()
        nn.init.ones_(self.context_norm.weight)
        
        self.patch_embed.reset_parameters()

        if self.config.use_rope:
            self.rope.reset_parameters()
        else:
            nn.init.zeros_(self.positional_embedding)
        
        # register tokens intialze with normal distribution
        nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        
        self.time_embed[0].reset_parameters()
        self.time_embed[2].reset_parameters()
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        for block in self.blocks:
            block.reset_parameters()
        
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        self.final_norm.reset_parameters()

    @apply_forward_hook
    def forward(self, x, context, context_attn_mask, timesteps):
        context = self.context_proj(context)
        context = self.context_norm(context)
        
        context_flat, context_cu_seqlens, context_max_seqlen_in_batch, context_indices = prepare_flash_attention_inputs(context, context_attn_mask)

        b, c, h, w = x.shape
        x = self.patch_embed(x)  # b, T, d

        x = torch.cat([self.register_tokens.repeat(b, 1, 1), x], 1)  # b, T + N, d

        if self.config.use_rope:
            cos, sin = self.rope(
                x,
                extend_with_register_tokens=16,
                height_width=(h // self.config.patch_size, w // self.config.patch_size),
            )
            cos = cos.repeat(1, b, 1)
            sin = sin.repeat(1, b, 1)
        else:
            x = x + self.positional_embedding.repeat(b, 1, 1)[:, : x.shape[1], :]
            cos, sin = None, None

        x_flat, x_cu_seqlens, x_max_seqlen_in_batch, x_indices = prepare_flash_attention_inputs(x)

        t_emb = timestep_embedding(timesteps * 1000, self.config.hidden_size).to(x.device, dtype=x.dtype)
        t_emb = self.time_embed(t_emb)
        modulation = self.adaLN_modulation(t_emb).repeat_interleave(
            16 + h // self.config.patch_size * w // self.config.patch_size, 
            dim=0
        ).chunk(9, dim=1)

        for _idx, block in enumerate(self.blocks):
            if self.config.gradient_checkpoint:
                x_flat = torch.utils.checkpoint.checkpoint(
                    block,
                    x_flat, x_cu_seqlens, x_max_seqlen_in_batch,
                    context_flat, context_cu_seqlens, context_max_seqlen_in_batch,
                    modulation,
                    (cos, sin),
                    use_reentrant=False,
                )
            else:
                x_flat = block(
                    x_flat, x_cu_seqlens, x_max_seqlen_in_batch, 
                    context_flat, context_cu_seqlens, context_max_seqlen_in_batch, 
                    modulation, (cos, sin)
                )

        x = unprepare_flash_attention_outputs(x_flat, x_indices, b, x_max_seqlen_in_batch, self.config.hidden_size)

        x = x[:, 16:, :]
        final_shift, final_scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + final_scale[:, None, :]) + final_shift[:, None, :]
        x = self.final_proj(x)

        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.config.patch_size,
            w=w // self.config.patch_size,
            p1=self.config.patch_size,
            p2=self.config.patch_size,
        )
        return x


if __name__ == "__main__":
    model = DiT(
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        train_bias_and_rms=True,
        use_rope=True,
    ).cuda()
    print(
        model(
            torch.randn(1, 4, 64, 64).cuda(),
            torch.randn(1, 37, 128).cuda(),
            torch.tensor([1.0]).cuda(),
        )
    )