# OpenCLIP Refactoring Guide

This guide documents the recent refactoring to support FSDP2 training, custom model loading from transformers, and 1-n alignment.

## Table of Contents

1. [Base Classes for Custom Encoders](#base-classes-for-custom-encoders)
2. [FSDP2 Training Support](#fsdp2-training-support)
3. [1-N Alignment Support](#1-n-alignment-support)
4. [Examples](#examples)

---

## Base Classes for Custom Encoders

### Overview

The refactoring introduces abstract base classes for vision and text encoders, making it easy to integrate custom models from transformers or other sources.

### Base Classes

#### `BaseVisionEncoder`

Located in `src/open_clip/base_model.py`

```python
from open_clip.base_model import BaseVisionEncoder

class MyCustomVisionEncoder(BaseVisionEncoder):
    def __init__(self, output_dim, image_size=224):
        super().__init__(output_dim, image_size)
        # Your initialization here

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        # return: (batch_size, output_dim) or ((batch_size, output_dim), tokens)
        pass
```

**Required Methods:**
- `forward(x)`: Takes images, returns embeddings (and optionally tokens)

**Optional Methods:**
- `lock(unlocked_groups=0, freeze_bn_stats=False)`: Freeze parameters
- `set_grad_checkpointing(enable=True)`: Enable gradient checkpointing
- `get_num_layers()`: Return number of layers

#### `BaseTextEncoder`

```python
from open_clip.base_model import BaseTextEncoder

class MyCustomTextEncoder(BaseTextEncoder):
    def __init__(self, output_dim, vocab_size=0, context_length=77):
        super().__init__(output_dim, vocab_size, context_length)
        # Your initialization here

    def forward(self, x):
        # x: (batch_size, sequence_length)
        # return: (batch_size, output_dim) or ((batch_size, output_dim), tokens)
        pass
```

### Pre-built Transformer Adapters

#### `TransformersVisionEncoder`

Wraps any HuggingFace vision model:

```python
from open_clip.base_model import TransformersVisionEncoder

# Use any HF vision model
encoder = TransformersVisionEncoder(
    model_name="google/vit-base-patch16-224",
    output_dim=512,
    image_size=224,
    pooler_type='cls',  # 'cls', 'mean', 'max'
    proj_type='linear',  # 'linear', 'mlp', 'none'
    pretrained=True,
)

# Forward pass
image_features = encoder(images)  # (batch_size, 512)
```

#### `TransformersTextEncoder`

Wraps any HuggingFace text model:

```python
from open_clip.base_model import TransformersTextEncoder

# Use any HF text model
encoder = TransformersTextEncoder(
    model_name="bert-base-uncased",
    output_dim=512,
    pooler_type='cls',  # 'cls', 'mean', 'max'
    proj_type='linear',  # 'linear', 'mlp', 'none'
    pretrained=True,
)

# Forward pass
text_features = encoder(text_tokens)  # (batch_size, 512)
```

### Updated Existing Encoders

The existing encoders now inherit from base classes:

- `TimmModel` → inherits from `BaseVisionEncoder`
- `HFTextEncoder` → inherits from `BaseTextEncoder`

This provides a consistent interface across all encoders.

---

## FSDP2 Training Support

### Overview

FSDP2 (Fully Sharded Data Parallel 2) provides improved memory efficiency and scalability for large-scale distributed training compared to DDP.

### Key Features

- Automatic parameter sharding across GPUs
- Mixed precision training (FP16, BF16)
- Activation checkpointing for memory efficiency
- Flexible wrapping policies for different model architectures

### Usage

#### Basic FSDP2 Wrapping

```python
from open_clip_train.fsdp2_utils import wrap_model_with_fsdp2

# Wrap entire model
model = wrap_model_with_fsdp2(
    model,
    mp_policy='bf16',  # 'fp16', 'bf16', 'fp32', or None
    reshard_after_forward=True,
)
```

#### Separate Vision/Text Encoder Wrapping

For asymmetric architectures, wrap encoders independently:

```python
from open_clip_train.fsdp2_utils import wrap_vision_text_encoders_fsdp2

model = wrap_vision_text_encoders_fsdp2(
    model,
    vision_encoder_name='visual',
    text_encoder_name='text',
    mp_policy='bf16',
)
```

#### Activation Checkpointing

Reduce memory usage by checkpointing activations:

```python
from open_clip_train.fsdp2_utils import setup_activation_checkpointing

# Apply to transformer blocks
model = setup_activation_checkpointing(model)
```

#### Mixed Precision Policy

```python
from open_clip_train.fsdp2_utils import setup_fsdp2_mixed_precision
import torch

mp_policy = setup_fsdp2_mixed_precision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.float32,
)

model = wrap_model_with_fsdp2(model, mixed_precision_policy=mp_policy)
```

#### Checkpoint Saving/Loading

```python
from open_clip_train.fsdp2_utils import get_fsdp2_state_dict, load_fsdp2_checkpoint

# Save checkpoint (gathers full state dict on rank 0)
if rank == 0:
    state_dict = get_fsdp2_state_dict(model, full_state_dict=True)
    torch.save(state_dict, 'checkpoint.pt')

# Load checkpoint
load_fsdp2_checkpoint(model, 'checkpoint.pt')
```

#### Integration with Training Script

```python
import torch
from open_clip_train.fsdp2_utils import (
    is_fsdp2_available,
    wrap_model_with_fsdp2,
    setup_activation_checkpointing,
    log_fsdp2_memory_stats,
)

# Check availability
if not is_fsdp2_available():
    raise RuntimeError("FSDP2 not available. Update PyTorch to >=2.0")

# Wrap model
model = wrap_model_with_fsdp2(
    model,
    mp_policy='bf16',
    reshard_after_forward=True,
)

# Optional: activation checkpointing
if args.grad_checkpointing:
    model = setup_activation_checkpointing(model)

# Training loop
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

    # Log memory stats periodically
    if step % 100 == 0:
        log_fsdp2_memory_stats(rank=dist.get_rank())
```

---

## 1-N Alignment Support

### Overview

Support for one-to-many alignment enables:
1. **1 text → N vision**: One text aligned with multiple vision encoders
2. **1 vision → N text**: One vision aligned with multiple text encoders
3. **M vision → N text**: Multiple encoders on both sides

### Multi-Encoder Models

#### `MultiVisionCLIP`

One text encoder with multiple vision encoders:

```python
from open_clip.multi_encoder_model import MultiVisionCLIP
from open_clip.timm_model import TimmModel
from open_clip.transformer import TextTransformer

# Create vision encoders
vision_encoders = [
    TimmModel('vit_base_patch16_224', embed_dim=512),
    TimmModel('convnext_base', embed_dim=512),
    TimmModel('resnet50', embed_dim=512),
]

# Create text encoder
text_encoder = TextTransformer(
    context_length=77,
    vocab_size=49408,
    width=512,
    heads=8,
    layers=12,
)

# Create multi-vision model
model = MultiVisionCLIP(
    vision_encoders=vision_encoders,
    text_encoder=text_encoder,
    embed_dim=512,
    output_dict=True,
)

# Forward pass
outputs = model(images, text_tokens)
# outputs = {
#     'text_features': (batch_size, 512),
#     'vision_features_list': [(batch_size, 512), (batch_size, 512), (batch_size, 512)],
#     'logit_scale': scalar
# }
```

#### `MultiTextCLIP`

One vision encoder with multiple text encoders:

```python
from open_clip.multi_encoder_model import MultiTextCLIP
from open_clip.base_model import TransformersTextEncoder

# Create text encoders
text_encoders = [
    TransformersTextEncoder('bert-base-uncased', output_dim=512),
    TransformersTextEncoder('roberta-base', output_dim=512),
    TransformersTextEncoder('distilbert-base-uncased', output_dim=512),
]

# Create vision encoder
vision_encoder = TimmModel('vit_base_patch16_224', embed_dim=512)

# Create multi-text model
model = MultiTextCLIP(
    vision_encoder=vision_encoder,
    text_encoders=text_encoders,
    embed_dim=512,
)

# Forward pass
outputs = model(images, text_tokens)
# outputs = {
#     'vision_features': (batch_size, 512),
#     'text_features_list': [(batch_size, 512), (batch_size, 512), (batch_size, 512)],
#     'logit_scale': scalar
# }
```

#### `MultiEncoderCLIP`

Multiple encoders on both sides (M:N alignment):

```python
from open_clip.multi_encoder_model import MultiEncoderCLIP

model = MultiEncoderCLIP(
    vision_encoders=vision_encoders,  # List of M vision encoders
    text_encoders=text_encoders,      # List of N text encoders
    embed_dim=512,
)
```

### One-to-Many Loss Functions

#### `OneToManyClipLoss`

Handles 1:N alignment with feature aggregation:

```python
from open_clip.one_to_many_loss import OneToManyClipLoss

# For 1 text with N vision encoders
loss_fn = OneToManyClipLoss(
    text_to_multi_vision=True,
    num_multi_encoders=3,
    aggregation='mean',  # 'mean', 'max', 'weighted'
    learnable_weights=False,  # Set True for learnable weighted aggregation
    rank=rank,
    world_size=world_size,
)

# Compute loss
text_features = model.encode_text(text_tokens)
vision_features_list = model.encode_image(images)

loss = loss_fn(
    anchor_features=text_features,
    multi_features_list=vision_features_list,
    logit_scale=model.logit_scale.exp(),
)
```

**Aggregation Strategies:**

- `mean`: Average all features
- `max`: Take max across all features
- `weighted`: Weighted sum (use `learnable_weights=True` to learn weights)

#### `MultiEncoderClipLoss`

For M:N alignment:

```python
from open_clip.one_to_many_loss import MultiEncoderClipLoss

loss_fn = MultiEncoderClipLoss(
    num_vision_encoders=3,
    num_text_encoders=2,
    aggregation='mean',
    rank=rank,
    world_size=world_size,
)

# Compute loss
vision_features_list = [v_enc(images) for v_enc in vision_encoders]
text_features_list = [t_enc(text) for t_enc in text_encoders]

loss = loss_fn(
    vision_features_list=vision_features_list,
    text_features_list=text_features_list,
    logit_scale=logit_scale,
)
```

---

## Examples

### Example 1: Custom Vision Encoder from HuggingFace

```python
from open_clip.base_model import TransformersVisionEncoder, TransformersTextEncoder
from open_clip.model import CLIP
import torch.nn as nn

# Use DINO v2 as vision encoder
vision_encoder = TransformersVisionEncoder(
    model_name="facebook/dinov2-base",
    output_dim=512,
    image_size=224,
    pooler_type='cls',
    proj_type='linear',
    pretrained=True,
)

# Use BERT as text encoder
text_encoder = TransformersTextEncoder(
    model_name="bert-base-uncased",
    output_dim=512,
    pooler_type='mean',
    proj_type='linear',
    pretrained=True,
)

# Create CLIP model (you would need to adapt CLIP class to accept these encoders)
# Or use them directly with OneToManyClipLoss
```

### Example 2: Multi-View Training with FSDP2

```python
import torch
from torch.utils.data import DataLoader
from open_clip.multi_encoder_model import MultiVisionCLIP
from open_clip.one_to_many_loss import OneToManyClipLoss
from open_clip_train.fsdp2_utils import wrap_model_with_fsdp2

# Create multi-view model
model = MultiVisionCLIP(
    vision_encoders=[
        TimmModel('vit_base_patch16_224', 512),
        TimmModel('vit_base_patch16_384', 512),
    ],
    text_encoder=text_encoder,
    embed_dim=512,
)

# Wrap with FSDP2
model = wrap_model_with_fsdp2(model, mp_policy='bf16')

# Create loss
loss_fn = OneToManyClipLoss(
    text_to_multi_vision=True,
    num_multi_encoders=2,
    aggregation='mean',
)

# Training loop
for images, texts in dataloader:
    outputs = model(images, texts)

    loss = loss_fn(
        anchor_features=outputs['text_features'],
        multi_features_list=outputs['vision_features_list'],
        logit_scale=outputs['logit_scale'],
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Example 3: Ensemble of Text Encoders

```python
from open_clip.multi_encoder_model import MultiTextCLIP
from open_clip.one_to_many_loss import OneToManyClipLoss

# Create ensemble of text encoders
text_encoders = [
    TransformersTextEncoder('bert-base-uncased', 512),
    TransformersTextEncoder('roberta-base', 512),
]

model = MultiTextCLIP(
    vision_encoder=TimmModel('vit_base_patch16_224', 512),
    text_encoders=text_encoders,
    embed_dim=512,
)

# Loss for 1 vision with N texts
loss_fn = OneToManyClipLoss(
    text_to_multi_vision=False,  # Vision is anchor
    num_multi_encoders=2,
    aggregation='weighted',
    learnable_weights=True,  # Learn optimal ensemble weights
)

# Training
outputs = model(images, texts)
loss = loss_fn(
    anchor_features=outputs['vision_features'],
    multi_features_list=outputs['text_features_list'],
    logit_scale=outputs['logit_scale'],
)
```

### Example 4: Full M:N Alignment

```python
from open_clip.multi_encoder_model import MultiEncoderCLIP
from open_clip.one_to_many_loss import MultiEncoderClipLoss

# 3 vision encoders × 2 text encoders
model = MultiEncoderCLIP(
    vision_encoders=[
        TimmModel('vit_base_patch16_224', 512),
        TimmModel('convnext_base', 512),
        TimmModel('resnet50', 512),
    ],
    text_encoders=[
        TransformersTextEncoder('bert-base-uncased', 512),
        TransformersTextEncoder('roberta-base', 512),
    ],
    embed_dim=512,
)

loss_fn = MultiEncoderClipLoss(
    num_vision_encoders=3,
    num_text_encoders=2,
    aggregation='mean',
)

# Training
outputs = model(images, texts)
loss = loss_fn(
    vision_features_list=outputs['vision_features_list'],
    text_features_list=outputs['text_features_list'],
    logit_scale=outputs['logit_scale'],
)
```

---

## Migration Guide

### Migrating from DDP to FSDP2

**Before (DDP):**
```python
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[device],
)
```

**After (FSDP2):**
```python
from open_clip_train.fsdp2_utils import wrap_model_with_fsdp2

model = wrap_model_with_fsdp2(
    model,
    mp_policy='bf16',
    reshard_after_forward=True,
)
```

### Creating Custom Encoders

**Before:**
```python
class MyEncoder(nn.Module):
    def __init__(self):
        # Custom implementation
        pass
```

**After:**
```python
from open_clip.base_model import BaseVisionEncoder

class MyEncoder(BaseVisionEncoder):
    def __init__(self, output_dim, image_size=224):
        super().__init__(output_dim, image_size)
        # Custom implementation
```

---

## API Reference

### Base Classes

- `BaseVisionEncoder`: Abstract base for vision encoders
- `BaseTextEncoder`: Abstract base for text encoders
- `TransformersVisionEncoder`: HF vision model adapter
- `TransformersTextEncoder`: HF text model adapter

### Multi-Encoder Models

- `MultiVisionCLIP`: 1 text → N vision
- `MultiTextCLIP`: 1 vision → N text
- `MultiEncoderCLIP`: M vision → N text

### Loss Functions

- `OneToManyClipLoss`: Handles 1:N alignment
- `MultiEncoderClipLoss`: Handles M:N alignment

### FSDP2 Utilities

- `wrap_model_with_fsdp2()`: Wrap model with FSDP2
- `wrap_vision_text_encoders_fsdp2()`: Wrap encoders separately
- `setup_activation_checkpointing()`: Enable activation checkpointing
- `get_fsdp2_state_dict()`: Get state dict from FSDP2 model
- `load_fsdp2_checkpoint()`: Load checkpoint into FSDP2 model
- `is_fsdp2_available()`: Check FSDP2 availability

---

## Performance Considerations

### FSDP2 vs DDP

- **Memory**: FSDP2 uses ~40-60% less memory than DDP for large models
- **Speed**: Slightly slower per step, but enables larger batch sizes
- **Scalability**: Better scaling to 100+ GPUs

### Multi-Encoder Overhead

- Each additional encoder adds computational cost
- Use `aggregation='mean'` for fastest training
- Consider freezing some encoders with `lock()` method

### Best Practices

1. **Start small**: Test with 1-2 encoders before scaling
2. **Use mixed precision**: BF16 recommended for modern GPUs
3. **Enable gradient checkpointing**: For memory-constrained setups
4. **Profile first**: Use `log_fsdp2_memory_stats()` to monitor memory

---

## Troubleshooting

### FSDP2 not available
```python
if not is_fsdp2_available():
    print("Update PyTorch to >= 2.0 for FSDP2 support")
```

### Out of memory
- Enable activation checkpointing
- Reduce batch size
- Use FP16/BF16 mixed precision
- Increase number of GPUs

### Slow training
- Check if `reshard_after_forward=True` (trades compute for memory)
- Verify mixed precision is enabled
- Ensure proper data loading (prefetch, num_workers)

---

## Contributing

When adding new encoder types:

1. Inherit from `BaseVisionEncoder` or `BaseTextEncoder`
2. Implement required `forward()` method
3. Optionally implement `lock()` and `set_grad_checkpointing()`
4. Add tests and documentation

---

## License

This refactoring maintains the original OpenCLIP license.
