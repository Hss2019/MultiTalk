# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp.cpu_offload import CPUOffload
from torch.distributed.utils import _free_storage

__all__ = ['shard_model']


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    # This policy defines which modules to wrap with FSDP.
    # Here, we wrap the 'blocks' of the model.
    auto_wrap_policy = partial(
        lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks
    )

    # Configure FSDP with CPU Offloading enabled.
    # This is the key to solving the initialization OOM.
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=device_id,
        sync_module_states=sync_module_states,
    )
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
