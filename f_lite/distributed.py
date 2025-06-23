# To heck with 🤗 accelerate

from functools import lru_cache
from typing import List, Tuple, Union

import torch
from torch import nn
from torch import distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def dist_max(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.MAX, group=mesh.get_group() if mesh else None)
    return tensor


def dist_mean(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.AVG, group=mesh.get_group() if mesh else None)
    return tensor


@lru_cache
def get_local_rank() -> int:
    return dist.get_rank() % torch.cuda.device_count()


@lru_cache
def get_global_rank() -> int:
    return dist.get_rank()


@lru_cache
def get_world_size() -> int:
    return dist.get_world_size()


@lru_cache
def is_main_process() -> bool:
    return get_global_rank() == 0


def get_device_mesh_hybrid_sharding():
    dp_shard = torch.cuda.device_count()
    world_size = get_world_size()
    dp_replicate = world_size // dp_shard
    assert (
        dp_replicate * dp_shard == world_size
    ), f"dp_replicate * dp_shard ({dp_replicate} * {dp_shard}) != world_size ({world_size})"

    dims = []
    names = []

    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("replicate")
    
    if dp_shard > 1:
        dims.append(dp_shard)
        names.append("shard")
    
    dims = tuple(dims)
    names = tuple(names)

    return init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)


def setup_torch_distributed(
        allow_tf32: bool=True,
        allow_bf16_reduced_precision_reduction: bool=True,
        detect_anomaly: bool=False,
):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(get_local_rank())
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16_reduced_precision_reduction
    torch.autograd.set_detect_anomaly(detect_anomaly)


def parallelize_model(
        model: torch.nn.Module, 
        device_mesh: DeviceMesh,
        param_dtype: torch.dtype,
        fsdp_grouping_plan: List[Tuple[nn.Module, bool]],
):
    fsdp_config = dict(
        mp_policy=MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
        ),
        mesh=device_mesh,
    )
    for module, reshard_after_forward in fsdp_grouping_plan:
        fully_shard(module, **fsdp_config, reshard_after_forward=reshard_after_forward)
    fully_shard(model, **fsdp_config, reshard_after_forward=True)

    return model
