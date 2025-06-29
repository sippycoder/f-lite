import argparse
import datetime
import logging
import math
import os
import random
import shutil
import time
import wandb

import numpy as np
import torch
import torch.utils.checkpoint
from torch.distributed._tensor import DTensor
import torch.distributed.checkpoint as dcp
from einops import rearrange
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_wsd_schedule
)

from f_lite.checkpoint import Checkpointer
from f_lite.data import ImageDataset
from f_lite.distributed import get_device_mesh_hybrid_sharding, get_global_rank, get_local_rank, get_world_size, is_main_process, parallelize_model, setup_torch_distributed
from f_lite.pipeline import FLitePipeline
from f_lite.precomputed_utils import (
    create_precomputed_data_loader,
    forward_with_precomputed_data,
)
from diffusers import AutoencoderKL
from transformers import Qwen2_5_VLModel, AutoProcessor
from f_lite.model import DiT
from f_lite.sampler import ResolutionBucketSampler, StatefulDistributedSampler
from f_lite.utils import make_image_grid

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
except ImportError:
    apply_liger_kernel_to_qwen2_5_vl = None

from dotenv import load_dotenv
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# Enable TF32 for faster training (only on NVIDIA Ampere or newer GPUs)
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="DiT (Diffusion Transformer) Fine-tuning Script")
    
    # Model parameters
    parser.add_argument("--pretrained_model_path", type=str, default=None, required=False,
                        help="Path to pretrained model")
    parser.add_argument("--vae_path", type=str, default=None, required=False,
                        help="Path to pretrained VAE")
    parser.add_argument("--text_encoder_path", type=str, default=None, required=False,
                        help="Path to pretrained text encoder")
    parser.add_argument("--processor_path", type=str, default=None, required=False,
                        help="Path to pretrained processor")
    parser.add_argument("--model_width", type=int, default=3072, 
                        help="Model width")
    parser.add_argument("--model_depth", type=int, default=40,
                        help="Model depth")
    parser.add_argument("--model_head_dim", type=int, default=256,
                        help="Attention head dimension")
    parser.add_argument("--rope_base", type=int, default=10_000,
                        help="Base for RoPE positional encoding")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=False,
                        help="Path to training dataset, supports CSV files or image directories")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation dataset, supports CSV files or image directories")
    parser.add_argument("--base_image_dir", type=str, default=None,
                        help="Base directory for image paths in CSV files")
    parser.add_argument("--image_column", type=str, default="image_path",
                        help="Column name in CSV containing image paths")
    parser.add_argument("--caption_column", type=str, default="caption",
                        help="Column name in CSV containing text captions")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Image resolution for training")
    parser.add_argument("--center_crop", action="store_true",
                        help="Whether to center crop images")
    parser.add_argument("--random_flip", action="store_true",
                        help="Whether to randomly flip images horizontally")
    parser.add_argument("--use_resolution_buckets", action="store_true",
                        help="Group images with same resolution into batches")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps, overrides num_epochs if provided")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient")
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        choices=["linear", "cosine", "wsd", "constant"],
                        help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether to use 8-bit Adam optimizer from bitsandbytes")
    parser.add_argument("--use_precomputed_data", action="store_true",
                help="Whether to use precomputed VAE latents and text embeddings")
    parser.add_argument("--precomputed_data_dir", type=str, default=None,
                    help="Directory containing precomputed data")

    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--train_only_lora", action="store_true",
                        help="Whether to freeze base model and train only LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, default="qkv,q,context_kv,proj",
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint to resume from")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="dit-finetuned",
                        help="Output directory for saving model and checkpoints")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help="Mixed precision training type")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Logging and evaluation parameters
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Logging directory")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "all"],
                        help="Logging integration to use")
    parser.add_argument("--project_name", type=str, default="dit-finetune",
                        help="Project name for wandb logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for the experiment")
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate sample images every X steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Run evaluation every X steps")
    parser.add_argument("--batch_multiplicity", type=int, default=1,
                        help="Repeat batch samples this many times")
    parser.add_argument("--sample_prompts_file", type=str, default=None,
                    help="Path to a text file containing prompts for sample image generation, one per line")
    
    return parser.parse_args()


def create_dataloader_with_dataset(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=4,
    seed=None,
    resolution=None,
    center_crop=False,
    random_flip=False,
    use_resolution_buckets=True,
    num_processes=1,
    process_index=0,
    sampler_state=None
):
    # Create sampler - either resolution bucket or standard
    if use_resolution_buckets:
        dataset.setup_aspect_ratio_buckets(min_side=resolution, max_ratio=1.0 if center_crop else 2.0)
        sampler = ResolutionBucketSampler(dataset, batch_size, shuffle=shuffle, num_replicas=num_processes, rank=process_index)
        if sampler_state:
            sampler.load_state_dict(sampler_state)
        data_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    else:
        # Standard approach
        sampler = StatefulDistributedSampler(dataset, batch_size=batch_size, num_replicas=num_processes, rank=process_index, shuffle=shuffle)
        if sampler_state:
            sampler.load_state_dict(sampler_state)
        logger.info(f"Using StatefulDistributedSampler rank:{process_index}/{num_processes} with {len(sampler)} batches")

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=4,
            pin_memory=True,
            drop_last=True,
        )
    
    return data_loader


def create_data_loader(
    data_path,
    batch_size,
    base_image_dir=None,
    shuffle=True,
    num_workers=4,
    seed=None,
    resolution=None,
    center_crop=False,
    random_flip=False,
    use_resolution_buckets=True,
    num_processes=1,
    process_index=0,
    sampler_state=None
):
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Create dataset
    dataset = ImageDataset(
        data_path=data_path,
        base_image_dir=base_image_dir,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        caption_column=args.caption_column,
        debug=args.debug
    )
    
    return create_dataloader_with_dataset(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        use_resolution_buckets=use_resolution_buckets,
        num_processes=num_processes,
        process_index=process_index,
        sampler_state=sampler_state
    )

def _convert_caption_to_messages(caption: str, metadata: dict, processor: AutoProcessor) -> str:
    system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. Generate images that are realistic and high-quality."
    if metadata is not None and metadata.get("media_type", "real") != "real":
        system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. The image doesn't need to be realistic, but it should be high-quality."

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Caption: \n\n{caption}"},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.no_grad()
def encode_prompt_with_qwen2_5_vl(
    text_encoder,
    processor,
    max_sequence_length=512,
    prompt=None,
    metadata=None,    # dict of metadata to add to the prompt
    num_images_per_prompt=1,
    device=None,
    return_index=-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # convert batch metadata {key: list of values} to metadata dictionary per prompt [{key: value}]
    if metadata is not None:
        metadata = [{k: v[i] for k, v in metadata.items()} for i in range(len(prompt))]
    else:
        metadata = [None] * len(prompt)

    messages = [_convert_caption_to_messages(p, m, processor) for p, m in zip(prompt, metadata)]

    inputs = processor(
        text=messages,
        padding="longest",
        pad_to_multiple_of=8,
        max_length=max_sequence_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device=device, dtype=text_encoder.dtype)
    attn_mask = inputs.attention_mask

    outputs = text_encoder(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
    prompt_embeds = outputs.hidden_states[return_index]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    if num_images_per_prompt > 1:
        # duplicate text embeddings and attention mask for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, attn_mask

def forward(
    dit_model,
    batch,
    vae_model,
    text_encoder,
    processor,
    device,
    global_step,
    master_process,
    generator=None,
    binnings=None,
    batch_multiplicity=None,
    bs_rampup=None,
    batch_size=None,
    return_index=-8,
):
    """
    Forward pass for DiT model.
    
    Args:
        dit_model: DiT model
        batch: Batch of data (images, metadata)
        vae_model: VAE model
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        device: Target device
        global_step: Current training step
        master_process: Whether this is the master process
        generator: Random number generator
        binnings: Tuple of (binning_dict, binning_count) for timestep binning
        batch_multiplicity: Factor to repeat batch
        bs_rampup: Steps for batch size rampup
        batch_size: Target batch size
        return_index: Index to return from encoded text
    """
    images_vae = batch.pop("image")

    # Get captions from metadata
    captions = batch.pop("caption")

    # Process images and captions
    # preprocess_start = time.time()
    images_vae = images_vae.to(device).to(torch.float32)
    
    with torch.no_grad():
        # Encode images using VAE
        vae_latent = vae_model.encode(images_vae).latent_dist.sample()
        # normalize
        vae_latent = (
            vae_latent - vae_model.config.shift_factor
        ) * vae_model.config.scaling_factor
        vae_latent = vae_latent.to(torch.bfloat16)
        
        # Encode captions
        caption_encoded, caption_attn_mask = encode_prompt_with_qwen2_5_vl(
            text_encoder,
            processor,
            prompt=captions,
            device=device,
            return_index=return_index,
        )
        caption_encoded = caption_encoded.to(torch.bfloat16)
    
    # Apply batch multiplicity if needed
    if batch_multiplicity is not None:
        # Repeat the batch by factor of batch_multiplicity
        vae_latent = vae_latent.repeat(batch_multiplicity, 1, 1, 1)
        caption_encoded = caption_encoded.repeat(batch_multiplicity, 1, 1)
        
    # Randomly zero out some caption_encoded (simulates classifier-free guidance)
    do_zero_out = torch.rand(caption_encoded.shape[0], device=device) < 0.05
    caption_encoded[do_zero_out] = 0
    caption_attn_mask[do_zero_out] = 1

    # Batch size rampup - gradually increase batch size during training
    if bs_rampup is not None and global_step < bs_rampup:
        target_bs = math.ceil((global_step + 1) * batch_size / bs_rampup / 4) * 4  # Round to multiple of 4
        if vae_latent.size(0) > target_bs:
            keep_indices = torch.randperm(vae_latent.size(0))[:target_bs]
            vae_latent = vae_latent[keep_indices]
            caption_encoded = caption_encoded[keep_indices]
    
    batch_size = vae_latent.size(0)
    
    # TIMEStep Sampling for diffusion process
    image_token_size = vae_latent.shape[2] * vae_latent.shape[3]  # h * w
    z = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    alpha = 2 * math.sqrt(image_token_size / (64 * 64))
    
    # Mix uniform and lognormal sampling for better coverage of timesteps
    do_uniform = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator) < 0.1
    uniform = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    t = torch.nn.Sigmoid()(z)
    lognormal = t * alpha / (1 + (alpha - 1) * t)
    
    # Final timesteps
    t = torch.where(~do_uniform, lognormal, uniform).to(torch.bfloat16)
    
    # Generate noise
    noise = torch.randn(
        vae_latent.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    
    # preprocess_time = time.time() - preprocess_start
    # if master_process:
    #     logger.debug(f"Preprocessing took {preprocess_time*1000:.2f}ms, alpha={alpha:.2f}")
    
    # Forward pass through DiT
    # forward_start = time.time()
    
    # Create noisy latents
    tr = t.reshape(batch_size, 1, 1, 1)
    z_t = vae_latent * (1 - tr) + noise * tr
    
    # Velocity prediction objective
    v_objective = vae_latent - noise
     
    # Forward through model
    output = dit_model(z_t, caption_encoded, caption_attn_mask, t)
    
    targ = rearrange(v_objective, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)
    pred = rearrange(output, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)

    diffusion_loss_batchwise = (
        (targ.float() - pred.float()).pow(2).mean(dim=(1, 2))
    )
    diffusion_loss = diffusion_loss_batchwise.mean()  
    
    # Combine losses
    total_loss = diffusion_loss
    
    # Record timestep binning statistics
    tbins = [min(int(_t * 10), 9) for _t in t]
    if binnings is not None:
        (
            diffusion_loss_binning,
            diffusion_loss_binning_count,
        ) = binnings
        for idx, tb in enumerate(tbins):
            diffusion_loss_binning[tb] += diffusion_loss_batchwise[idx].item()
            diffusion_loss_binning_count[tb] += 1
    
    # forward_time = time.time() - forward_start
    # if master_process:
    #     logger.debug(f"Forward pass took {forward_time*1000:.2f}ms")
    
    return total_loss, diffusion_loss

def sample_images(
    dit_model,
    vae_model,
    text_encoder,
    processor,
    device,
    global_step,
    prompts=None,
    image_width=512,
    image_height=512,
    prompts_per_gpu=1,
    num_inference_steps=50,
    cfg_scale=6.0,
    return_index=-8,  
    prompts_file=None,
): 
    if prompts_file is not None and os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                file_prompts = [line.strip() for line in f.readlines() if line.strip()]
            if file_prompts:
                logger.info(f"Using {len(file_prompts)} prompts from {prompts_file}")
                prompts = file_prompts
            else:
                logger.warning(f"Prompt file {prompts_file} is empty, using default prompts")
        except Exception as e:
            logger.error(f"Error reading prompts file: {e}. Using default prompts.")
    
    # use default prompts if none is provided
    if prompts is None:
        prompts = [
            "a beautiful photograph of a mountain landscape at sunset",
            "a cute cat playing with a ball of yarn",
            "a futuristic cityscape with flying cars",
            "an oil painting of a flower garden",
        ]
    
    logger.info(f"Generating {len(prompts)} sample images at step {global_step}")
    
    # Store and set model to eval mode
    previous_training_state = dit_model.training
    dit_model.eval()
    
    samples = []

    # Generate samples with correct sampling logic
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            # Encode prompt
            prompt_embeds, prompt_attn_mask = encode_prompt_with_qwen2_5_vl(
                text_encoder,
                processor,
                prompt=[prompt],
                device=device,
                return_index=return_index,
            )
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
            prompt_attn_mask = prompt_attn_mask.to(torch.bfloat16)
            
            # Create unconditional embeddings for CFG
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_attn_mask = torch.ones_like(prompt_attn_mask)
            
            # Initialize latents with correct channel count (16)
            batch_size = 1
            generator = torch.Generator(device=device).manual_seed(global_step + i)
            
            latent_height = image_height // 8
            latent_width = image_width // 8
            latent_shape = (batch_size, vae_model.config.latent_channels, latent_height, latent_width)
            latents = torch.randn(latent_shape, device=device, generator=generator, dtype=torch.bfloat16)
            
            # Setup timesteps with proper scaling
            image_token_size = latent_height * latent_width
            alpha = 2 * math.sqrt(image_token_size / (64 * 64))
            
            # Sampling loop
            for j in range(num_inference_steps, 0, -1):
                # Calculate current and next timesteps
                t = j / num_inference_steps
                t_next = (j - 1) / num_inference_steps
                
                # Apply alpha adjustment for non-uniform timesteps
                t = t * alpha / (1 + (alpha - 1) * t)   
                t_next = t_next * alpha / (1 + (alpha - 1) * t_next)
                dt = t - t_next
                
                t_tensor = torch.tensor([t] * batch_size).to(device, torch.bfloat16)
                
                # Get model prediction
                model_output = dit_model(latents.to(torch.bfloat16), prompt_embeds, prompt_attn_mask, t_tensor)
                
                # Apply classifier-free guidance
                if cfg_scale > 1:
                    uncond_output = dit_model(latents.to(torch.bfloat16), negative_embeds, negative_attn_mask, t_tensor)
                    model_output = uncond_output + cfg_scale * (model_output - uncond_output)
                
                # Update latents with simple velocity update
                latents = latents + dt * model_output.to(dtype=torch.float32)
            
            # Properly decode latents using VAE
            latents = latents / vae_model.config.scaling_factor + vae_model.config.shift_factor
            image = vae_model.decode(latents.to(torch.float32)).sample
            
            # Post-process images
            # VAE output is in [-1, 1] range. `save_image` with `normalize=True` will handle conversion.
            # The previous code was incorrectly converting to uint8 and permuting dimensions.
            samples.append(image.cpu()[0])

    image_grid = make_image_grid(samples, nrow=3, normalize=True, value_range=(-1, 1))
    
    # Restore original training state
    dit_model.train(previous_training_state)
    
    return image_grid


def build_fsdp_grouping_plan(model: DiT):
    group_plan = [(block, False) for block in model.blocks]
    return group_plan


def train(args):
    """
    Main training function.
    
    Args:
        args: Script arguments
    """
    setup_torch_distributed()
    device_mesh = get_device_mesh_hybrid_sharding()
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = f"cuda:{local_rank}"
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    
    # Set logger level for various components
    if is_main_process():
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        # Create a common seed for data loading
        common_seed = args.seed
    else:
        common_seed = random.randint(0, 1000000)
    
    logger.info(f"Using random seed: {common_seed}")

    torch.manual_seed(common_seed)
    
    # Initialize wandb if needed
    if args.report_to in ["wandb", "all"] and is_main_process():
        run_name = args.run_name if args.run_name else f"DiT-pretrain-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        wandb.init(
            project=args.project_name,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
    if args.pretrained_model_path is not None:
        logger.info(f"Loading model from {args.pretrained_model_path}")
        pipeline = FLitePipeline.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else None
        )
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
        text_encoder = Qwen2_5_VLModel.from_pretrained(args.text_encoder_path, torch_dtype=torch.bfloat16)
        if apply_liger_kernel_to_qwen2_5_vl is not None:
            apply_liger_kernel_to_qwen2_5_vl(text_encoder)
        processor = AutoProcessor.from_pretrained(args.processor_path)
        dit_model = DiT(
            in_channels=vae.config.latent_channels,
            patch_size=2,
            hidden_size=args.model_width,
            depth=args.model_depth,
            num_heads=args.model_width // args.model_head_dim,
            mlp_ratio=4.0,
            cross_attn_input_size=text_encoder.config.hidden_size,
            train_bias_and_rms=True,
            use_rope=True,
            gradient_checkpoint=args.gradient_checkpointing,
            dynamic_softmax_temperature=False,
            rope_base=args.rope_base,
        )
        pipeline = FLitePipeline(
            dit_model=dit_model,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor)

    # Load only the models needed based on precomputed status
    if args.use_precomputed_data:
        # Only load DiT model if using precomputed data
        dit_model = pipeline.dit_model
        
        # Set VAE and text_encoder to None to save memory
        vae_model = None
        text_encoder = None
        processor = None

        logger.info("Using precomputed data - VAE and text encoder not loaded to save memory")
    else:
        # Load all models
        dit_model = pipeline.dit_model
        vae_model = pipeline.vae
        text_encoder = pipeline.text_encoder
        processor = pipeline.processor

        # Move models to device
        vae_model = vae_model.to(device)
        text_encoder = text_encoder.to(device)

        # Freeze encoders (they are only used for inference)
        vae_model.requires_grad_(False)
        text_encoder.requires_grad_(False)

        logger.info("Compiling VAE...")
        vae_model = torch.compile(vae_model, mode='reduce-overhead')
    
    dit_model.train() 
    
    # Count parameters
    param_count = sum(p.numel() for p in dit_model.parameters())
    logger.info(f"Number of parameters: {param_count / 1e6:.2f} million")

    # Apply LoRA if specified
    if args.use_lora:
        logger.info("Setting up LoRA fine-tuning")

        # If only training LoRA weights, freeze base model
        if args.train_only_lora:
            dit_model.requires_grad_(False)
            logger.info("Freezing base model parameters, training only LoRA weights")

        # Parse target modules from comma-separated string
        target_modules = [module.strip() for module in args.lora_target_modules.split(",")]

        # Add LoRA adapters to the model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",  # Don't apply LoRA to bias terms
            init_lora_weights="gaussian"  # Initialize with Gaussian distribution
        )

        dit_model.add_adapter(lora_config)
        for name, param in dit_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        # Load LoRA checkpoint if provided
        if args.lora_checkpoint is not None:
            logger.info(f"Loading LoRA weights from {args.lora_checkpoint}")
            lora_state_dict = torch.load(args.lora_checkpoint, map_location=device)
            set_peft_model_state_dict(dit_model, lora_state_dict)

        # Log LoRA parameter count
        lora_param_count = sum(p.numel() for n, p in dit_model.named_parameters() if "lora" in n.lower() and p.requires_grad)
        trainable_params = sum(p.numel() for p in dit_model.parameters() if p.requires_grad)

        logger.info(f"Added LoRA adapter with rank {args.lora_rank}")
        logger.info(f"Target modules: {args.lora_target_modules}")
        logger.info(f"LoRA parameters: {lora_param_count / 1e6:.2f}M ({100 * lora_param_count / param_count:.2f}% of total)")
        logger.info(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")
    
    if args.use_precomputed_data:
        if args.precomputed_data_dir is None:
            raise ValueError("When using precomputed data, precomputed_data_dir must be specified")

        logger.info(f"Using precomputed data from {args.precomputed_data_dir}")
        train_dataloader = create_precomputed_data_loader(
            precomputed_data_dir=args.precomputed_data_dir,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
            random_flip=args.random_flip,
            use_resolution_buckets=args.use_resolution_buckets,
        )

        val_dataloader = None
        if args.val_data_path:
            # For validation, we can still use precomputed data if available
            val_data_dir = os.path.join(args.precomputed_data_dir, "validation") 
            if os.path.exists(val_data_dir):
                val_dataloader = create_precomputed_data_loader(
                    precomputed_data_dir=val_data_dir,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                )
            else:
                logger.warning(f"Validation precomputed data not found at {val_data_dir}. Skipping validation.")
                val_dataloader = None
    else:
        train_dataloader = create_data_loader(
            data_path=args.train_data_path,
            batch_size=args.train_batch_size,
            base_image_dir=args.base_image_dir,
            shuffle=True,
            num_workers=4,
            seed=common_seed,
            resolution=args.resolution,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            use_resolution_buckets=args.use_resolution_buckets,
            num_processes=world_size,
            process_index=global_rank,
        )

        val_dataloader = None
        if args.val_data_path:
            val_dataloader = create_data_loader(
                data_path=args.val_data_path,
                batch_size=args.eval_batch_size,
                base_image_dir=args.base_image_dir,
                shuffle=False,
                num_workers=4,
                num_processes=world_size,
                process_index=global_rank,
            )
    
    # Initialize optimizer
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer from bitsandbytes")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to regular AdamW")
    
    # Wrap the model in FSDP2
    dit_model = parallelize_model(
        model=dit_model, 
        device_mesh=device_mesh, 
        param_dtype=torch.bfloat16,
        fsdp_grouping_plan=build_fsdp_grouping_plan(dit_model),
    )

    optimizer = optimizer_class(
        dit_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        fused=True,
    )
    
    # Calculate number of steps
    if args.max_steps:
        max_steps = args.max_steps
    else:
        max_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    
    # Initialize learning rate scheduler
    if args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_steps,
        )
    elif args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_steps,
        )
    elif args.lr_scheduler == "wsd":
        lr_scheduler = get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_decay_steps=max_steps // 10,  # 10% of the training steps for decay
            num_stable_steps=max_steps - args.num_warmup_steps - max_steps // 10,  # 90% of the training steps for stable
        )
    else:  # constant with warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max_steps * 1000,  # Very long decay to simulate constant
        )

    if args.use_lora: 
        optimizer = optimizer_class(
            [p for p in dit_model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpointer = Checkpointer(args.output_dir, dcp_api=True)
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        last_training_time = None
        
        if checkpoint_path == "latest":
            last_training_time = checkpointer.last_training_time
            logger.info(f"Resuming from checkpoint at {last_training_time}")
            global_step = last_training_time
            checkpoint_path = None
        else:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            global_step = int(checkpoint_path.split("/")[-1])
    
        checkpointer.load_model(dit_model, last_training_time, checkpoint_path)
        checkpointer.load_optim(dit_model, optimizer, last_training_time, checkpoint_path)
        sampler_state = checkpointer.load_sampler_state(last_training_time, checkpoint_path)
        train_dataloader = create_dataloader_with_dataset(
            dataset=train_dataloader.dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
            resolution=args.resolution,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            use_resolution_buckets=args.use_resolution_buckets,
            num_processes=world_size,
            process_index=global_rank,
            sampler_state=sampler_state,
        )
    else:
        logger.info("No checkpoint found, starting from scratch")
        global_step = 0
        dit_model.to_empty(device=device)
        dit_model.reset_parameters()
    
    # Initialize training progress bar
    progress_bar = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not is_main_process(),
        desc="Training",
    )

    # Binning for loss analysis
    diffusion_loss_binning = {k: 0 for k in range(10)}
    diffusion_loss_binning_count = {k: 0 for k in range(10)}
    
    # Training loop
    dit_model.train()

    dataset = train_dataloader.dataset
    print(
        f"\nDataset size:       {len(dataset)} images"
        f"\nDataloader batches:   {len(train_dataloader)}"
        f"\nCalculated max steps: {len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps}"
        f"\nWorld size:           {get_world_size()}"
        f"\nLocal rank:           {get_local_rank()}"
        f"\nGlobal rank:          {get_global_rank()}"
        f"\nIs main process:      {is_main_process()}"
        f"\nDevice mesh:          {device_mesh}"
        f"\nDevice:               {device}"
        f"\nBatch size:           {args.train_batch_size}\n"
    )
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Set epoch for data sampler (only applicable for distributed training)
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Track time
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):    
            # Forward pass
            if args.use_precomputed_data:
                total_loss, diffusion_loss = forward_with_precomputed_data(
                    dit_model=dit_model,
                    batch=batch,
                    device=device,
                    global_step=global_step,
                    master_process=is_main_process(),
                    binnings=(diffusion_loss_binning, diffusion_loss_binning_count),
                    batch_multiplicity=args.batch_multiplicity,
                    bs_rampup=None,
                    batch_size=args.train_batch_size,
                )
            else:
                total_loss, diffusion_loss = forward(
                    dit_model=dit_model,
                    batch=batch,
                    vae_model=vae_model,
                    text_encoder=text_encoder,
                    processor=processor,
                    device=device,
                    global_step=global_step,
                    master_process=is_main_process(),
                    binnings=(diffusion_loss_binning, diffusion_loss_binning_count),
                    batch_multiplicity=args.batch_multiplicity,
                    bs_rampup=None,
                    batch_size=args.train_batch_size,
                    return_index=-8,
                )
                
            # Backward pass
            total_loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(dit_model.parameters(), args.max_grad_norm)

            # Update optimizer and scheduler
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                logs = {
                    "train/loss": total_loss.item(),
                    "train/diffusion_loss": diffusion_loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }

                if grad_norm is not None:
                    logs["train/grad_norm"] = (
                        grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                    ).item()

                # Log average bin loss
                for i in range(10):
                    if diffusion_loss_binning_count[i] > 0:
                        avg_bin_loss = (
                            diffusion_loss_binning[i]
                            / diffusion_loss_binning_count[i]
                        )
                        logs[f"metrics/avg_loss_bin_{i}"] = avg_bin_loss
                
                # Log bin counts as a histogram
                if any(diffusion_loss_binning_count.values()) and is_main_process():
                    raw_data = []
                    for bin_idx, count in diffusion_loss_binning_count.items():
                        raw_data.extend([bin_idx] * int(count))
                    wandb.log(
                        {"metrics/diffusion_loss_bin_counts": wandb.Histogram(raw_data)},
                        step=global_step,
                    )

                # Reset binning stats for next logging interval
                diffusion_loss_binning.clear()
                diffusion_loss_binning.update({k: 0 for k in range(10)})
                diffusion_loss_binning_count.clear()
                diffusion_loss_binning_count.update({k: 0 for k in range(10)})
                
                # Log to all trackers
                if is_main_process():
                    wandb.log(logs, step=global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "diff_loss": f"{diffusion_loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.7f}",
                }) 
            
            # Save checkpoint
            if global_step % args.checkpointing_steps == 0:
                sampler_state = train_dataloader.sampler.state_dict(global_step) if not args.use_resolution_buckets else train_dataloader.batch_sampler.state_dict(global_step)
                checkpointer.save(global_step, dit_model, optimizer, sampler_state)
                if is_main_process():
                    logger.info(f"Saved checkpoint to {args.output_dir}/dcp_api/{global_step}")

                    # Remove old checkpoints
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [
                            d
                            for d in os.listdir(args.output_dir)
                            if os.path.isdir(os.path.join(args.output_dir, d))
                        ]
                        # Sort by step number (ascending)
                        checkpoints.sort(key=lambda x: int(x))

                        if len(checkpoints) > args.checkpoints_total_limit:
                            num_to_delete = len(checkpoints) - args.checkpoints_total_limit
                            for ckpt_to_delete in checkpoints[:num_to_delete]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt_to_delete))
                                logger.info(f"Removed old checkpoint: {ckpt_to_delete}")

            # Sample images
            if global_step % args.sample_every == 0:
                # For sampling, we need VAE and text encoder
                temp_vae = None
                temp_text_encoder = None
                temp_tokenizer = None

                if args.use_precomputed_data:
                    # Temporarily load VAE and text encoder for sampling
                    logger.info("Temporarily loading VAE and text encoder for sampling")
                    temp_pipeline = FLitePipeline.from_pretrained(
                        args.pretrained_model_path,
                        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else None
                    )
                    temp_vae = temp_pipeline.vae.to(device)
                    temp_text_encoder = temp_pipeline.text_encoder.to(device)
                    temp_tokenizer = temp_pipeline.tokenizer

                    # Use temporary models for sampling
                    image_grid = sample_images(
                        dit_model=dit_model,
                        vae_model=temp_vae,
                        text_encoder=temp_text_encoder,
                        tokenizer=temp_tokenizer,
                        device=device,
                        global_step=global_step,
                        prompts=None,  # Use default prompts as fallback
                        image_width=384,
                        image_height=384,
                        prompts_per_gpu=2,
                        prompts_file=args.sample_prompts_file,
                    )

                    # Clean up temporary models
                    del temp_vae, temp_text_encoder, temp_tokenizer, temp_pipeline
                    torch.cuda.empty_cache()
                else:
                    # Use existing models
                    image_grid = sample_images(
                        dit_model=dit_model,
                        vae_model=vae_model,
                        text_encoder=text_encoder,
                        processor=processor,
                        device=device,
                        global_step=global_step,
                        prompts=None,  # Use default prompts as fallback
                        image_width=256,
                        image_height=256,
                        prompts_per_gpu=2,
                        prompts_file=args.sample_prompts_file,
                    )
            
                # Save image grid to wandb
                if is_main_process():
                    wandb.log({
                        "samples": [wandb.Image(image_grid, caption=f"Step {global_step}")],
                    }, step=global_step)
            
            # Run evaluation
            if val_dataloader and global_step % args.eval_every == 0:
                logger.info(f"Running evaluation at step {global_step}")
                dit_model.eval()
                
                val_loss = 0.0
                val_diffusion_loss = 0.0
                val_count = 0
                
                # Evaluate on validation set
                for val_step, val_batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        val_total_loss, val_diff_loss = forward(
                            dit_model=dit_model,
                            batch=val_batch,
                            vae_model=vae_model,
                            text_encoder=text_encoder,
                            processor=processor,
                            device=device,
                            global_step=global_step,
                            master_process=is_main_process(),
                            return_index=-8,
                        )
                        
                        val_loss += val_total_loss.item()
                        val_diffusion_loss += val_diff_loss.item()
                        val_count += 1
                        
                        # Limit validation to 20 batches
                        if val_step >= 19:
                            break
                
                # Calculate average validation loss
                if val_count > 0:
                    val_loss /= val_count
                    val_diffusion_loss /= val_count
                    
                    # Log validation results
                    if is_main_process():
                        logs = {
                            "val/loss": val_loss,
                            "val/diffusion_loss": val_diffusion_loss,
                        }
                        wandb.log(logs, step=global_step)
                        
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                
                # Set model back to training mode
                dit_model.train()
            
            # Check if we've reached max steps
            if global_step >= max_steps:
                logger.info(f"Reached max steps ({max_steps}). Stopping training.")
                break
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # Save model at the end of each epoch
        sampler_state = train_dataloader.sampler.state_dict(global_step) if not args.use_resolution_buckets else train_dataloader.batch_sampler.state_dict(global_step)
        checkpointer.save(global_step, dit_model, optimizer, sampler_state)
        if is_main_process():
            logger.info(f"Saved checkpoint to {args.output_dir}/dcp_api/{global_step}")
        
        # Check if we've reached max steps
        if global_step >= max_steps:
            break
    
    # End of training
    logger.info(f"Training completed after {global_step} steps!")
    
    # Save the final model
    sampler_state = train_dataloader.sampler.state_dict(global_step) if not args.use_resolution_buckets else train_dataloader.batch_sampler.state_dict(global_step)
    checkpointer.save(global_step, dit_model, optimizer, sampler_state)
    if is_main_process():
        final_model_path = os.path.join(args.output_dir, f"dcp_api/{global_step}")
        os.makedirs(final_model_path, exist_ok=True)
        
        logger.info(f"Final model saved to {final_model_path}")

        # Save LoRA weights separately if using LoRA
        if args.use_lora:
            lora_state_dict = get_peft_model_state_dict(dit_model)
            torch.save(
                lora_state_dict,
                os.path.join(final_model_path, "lora_weights.pt")
            )
            logger.info(f"Saved final LoRA weights to {final_model_path}/lora_weights.pt")
        

if __name__ == "__main__":
    args = parse_args()
    train(args)