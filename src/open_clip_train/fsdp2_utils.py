"""FSDP2 (Fully Sharded Data Parallel 2) utilities for distributed training.

This module provides utilities for wrapping models with FSDP2, which offers improved
memory efficiency and scalability compared to DDP for large-scale training.
"""
import functools
import logging
from typing import Optional, Callable, Union, List

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


def get_fsdp_wrap_policy(
    model_type: str = 'clip',
    transformer_layer_name: Optional[str] = None,
) -> Callable:
    """Get the FSDP wrapping policy for different model types.

    Args:
        model_type: Type of model ('clip', 'coca', 'custom')
        transformer_layer_name: Name of transformer layer class to wrap
            (e.g., 'ResidualAttentionBlock', 'Block')

    Returns:
        A callable that returns True for modules that should be wrapped
    """
    def clip_auto_wrap_policy(module: nn.Module, recurse: bool, **kwargs) -> bool:
        """Auto-wrap policy for CLIP models.

        Wraps each transformer block separately for optimal sharding.
        """
        from open_clip.transformer import ResidualAttentionBlock, Transformer
        from open_clip.model import CLIP

        if recurse:
            return True

        # Wrap transformer blocks
        return isinstance(module, (ResidualAttentionBlock, Transformer))

    def transformer_auto_wrap_policy(module: nn.Module, recurse: bool, layer_cls) -> bool:
        """Generic transformer wrapping policy."""
        if recurse:
            return True
        return isinstance(module, layer_cls)

    if model_type == 'clip':
        return clip_auto_wrap_policy
    elif transformer_layer_name:
        # Custom policy based on layer name
        try:
            # Try to import the layer class dynamically
            parts = transformer_layer_name.rsplit('.', 1)
            if len(parts) == 2:
                module_name, class_name = parts
                import importlib
                mod = importlib.import_module(module_name)
                layer_cls = getattr(mod, class_name)
            else:
                # Assume it's a class in open_clip.transformer
                from open_clip import transformer as trans_module
                layer_cls = getattr(trans_module, transformer_layer_name)

            return functools.partial(transformer_auto_wrap_policy, layer_cls=layer_cls)
        except (ImportError, AttributeError) as e:
            logging.warning(f"Could not load transformer layer {transformer_layer_name}: {e}")
            return clip_auto_wrap_policy
    else:
        return clip_auto_wrap_policy


def setup_fsdp2_mixed_precision(
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    buffer_dtype: Optional[torch.dtype] = None,
) -> MixedPrecisionPolicy:
    """Setup mixed precision policy for FSDP2.

    Args:
        param_dtype: Data type for parameters (e.g., torch.float32, torch.bfloat16)
        reduce_dtype: Data type for gradient reduction
        buffer_dtype: Data type for buffers

    Returns:
        MixedPrecisionPolicy configured with specified dtypes
    """
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )


def wrap_model_with_fsdp2(
    model: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    mixed_precision_policy: Optional[MixedPrecisionPolicy] = None,
    reshard_after_forward: bool = True,
    mp_policy: Optional[str] = None,
) -> nn.Module:
    """Wrap a model with FSDP2 for distributed training.

    Args:
        model: The model to wrap
        device_mesh: Optional DeviceMesh for multi-dimensional parallelism
        mixed_precision_policy: Optional mixed precision policy
        reshard_after_forward: Whether to reshard parameters after forward pass
        mp_policy: Mixed precision policy name ('fp16', 'bf16', 'fp32', or None)

    Returns:
        The FSDP2-wrapped model
    """
    # Setup mixed precision if specified
    if mp_policy and mixed_precision_policy is None:
        if mp_policy == 'bf16':
            mixed_precision_policy = setup_fsdp2_mixed_precision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
        elif mp_policy == 'fp16':
            mixed_precision_policy = setup_fsdp2_mixed_precision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
            )
        # fp32 or None means no mixed precision

    # Apply FSDP2 to the model
    # FSDP2 uses fully_shard which is a composable API
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mixed_precision_policy,
        reshard_after_forward=reshard_after_forward,
    )

    return model


def wrap_vision_text_encoders_fsdp2(
    model: nn.Module,
    vision_encoder_name: str = 'visual',
    text_encoder_name: str = 'text',
    device_mesh: Optional[DeviceMesh] = None,
    mixed_precision_policy: Optional[MixedPrecisionPolicy] = None,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Wrap vision and text encoders separately with FSDP2.

    This allows for independent sharding of vision and text encoders,
    which can be beneficial for models with asymmetric architectures.

    Args:
        model: The CLIP model to wrap
        vision_encoder_name: Attribute name of vision encoder in model
        text_encoder_name: Attribute name of text encoder in model
        device_mesh: Optional DeviceMesh for multi-dimensional parallelism
        mixed_precision_policy: Optional mixed precision policy
        reshard_after_forward: Whether to reshard parameters after forward pass

    Returns:
        The FSDP2-wrapped model
    """
    # Wrap vision encoder
    if hasattr(model, vision_encoder_name):
        vision_encoder = getattr(model, vision_encoder_name)
        fully_shard(
            vision_encoder,
            mesh=device_mesh,
            mp_policy=mixed_precision_policy,
            reshard_after_forward=reshard_after_forward,
        )

    # Wrap text encoder
    if hasattr(model, text_encoder_name):
        text_encoder = getattr(model, text_encoder_name)
        fully_shard(
            text_encoder,
            mesh=device_mesh,
            mp_policy=mixed_precision_policy,
            reshard_after_forward=reshard_after_forward,
        )

    # Wrap the entire model for the final projection layers
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mixed_precision_policy,
        reshard_after_forward=reshard_after_forward,
    )

    return model


def setup_activation_checkpointing(
    model: nn.Module,
    checkpoint_wrapper_fn: Optional[Callable] = None,
    auto_wrap_policy: Optional[Callable] = None,
) -> nn.Module:
    """Setup activation checkpointing for memory efficiency.

    Args:
        model: The model to apply activation checkpointing to
        checkpoint_wrapper_fn: Optional custom checkpoint wrapper function
        auto_wrap_policy: Policy to determine which modules to checkpoint

    Returns:
        Model with activation checkpointing applied
    """
    if checkpoint_wrapper_fn is None:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

    if auto_wrap_policy is None:
        # Default: checkpoint transformer blocks
        from open_clip.transformer import ResidualAttentionBlock
        auto_wrap_policy = lambda module: isinstance(module, ResidualAttentionBlock)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
        check_fn=auto_wrap_policy,
    )

    return model


def get_fsdp2_state_dict(model: nn.Module, full_state_dict: bool = True) -> dict:
    """Get state dict from FSDP2 model.

    Args:
        model: FSDP2-wrapped model
        full_state_dict: Whether to gather full state dict on rank 0

    Returns:
        State dict
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    if full_state_dict:
        # Use FSDP's state_dict utilities to gather full state on rank 0
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()
    else:
        # Return local sharded state dict
        return model.state_dict()


def load_fsdp2_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
) -> None:
    """Load checkpoint into FSDP2 model.

    Args:
        model: FSDP2-wrapped model
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict if checkpoint contains other info
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Use FSDP's load_state_dict
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        model.load_state_dict(state_dict, strict=strict)


def is_fsdp2_available() -> bool:
    """Check if FSDP2 is available in current PyTorch version.

    Returns:
        True if FSDP2 is available, False otherwise
    """
    try:
        from torch.distributed._composable.fsdp import fully_shard
        return True
    except ImportError:
        return False


def log_fsdp2_memory_stats(rank: int = 0):
    """Log FSDP2 memory statistics.

    Args:
        rank: Process rank
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logging.info(
            f"[Rank {rank}] GPU Memory - "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Max Allocated: {max_allocated:.2f}GB"
        )
