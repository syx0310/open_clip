# DinoV3 + Qwen Embedding Training Guide

Complete guide for training CLIP with DinoV3 vision and Qwen text encoder using transformers.

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to the open_clip directory
cd /home/user/open_clip

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow
pip install -e .
```

### 2. Run the Training Script

```bash
# Simple 1:1 training (DinoV3 + Qwen)
python examples/train_dinov3_qwen.py

# With custom arguments
python examples/train_dinov3_qwen.py \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --device cuda
```

## Supported Configurations

### 1:1 Alignment (Standard CLIP)
- 1 DinoV3 vision encoder + 1 Qwen text encoder
```bash
python examples/train_dinov3_qwen.py --mode 1to1
```

### 1:N Alignment (1 Text with N Vision)
- 1 Qwen text encoder + Multiple DinoV3 vision encoders
```bash
python examples/train_dinov3_qwen.py --mode 1text_nvision --num-vision 3
```

### N:1 Alignment (N Text with 1 Vision)
- Multiple Qwen text encoders + 1 DinoV3 vision encoder
```bash
python examples/train_dinov3_qwen.py --mode ntext_1vision --num-text 2
```

## Model Options

### Vision Models (DinoV3)
- `facebook/dinov2-small` - 22M params
- `facebook/dinov2-base` - 86M params (recommended)
- `facebook/dinov2-large` - 304M params
- `facebook/dinov2-giant` - 1.1B params

### Text Models (Qwen)
- `Qwen/Qwen2-0.5B` - 0.5B params (for testing)
- `Qwen/Qwen2-1.5B` - 1.5B params
- `Qwen/Qwen2-7B` - 7B params

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30.0
- CUDA (recommended for GPU training)

## Memory Requirements

| Configuration | GPU Memory | Batch Size |
|--------------|------------|------------|
| DinoV2-Base + Qwen2-0.5B | ~8GB | 32 |
| DinoV2-Large + Qwen2-1.5B | ~16GB | 16 |
| DinoV2-Giant + Qwen2-7B | ~40GB | 8 |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python examples/train_dinov3_qwen.py --batch-size 8

# Use gradient checkpointing
python examples/train_dinov3_qwen.py --gradient-checkpointing

# Use mixed precision
python examples/train_dinov3_qwen.py --mixed-precision bf16
```

### Slow Training
```bash
# Enable torch.compile (PyTorch 2.0+)
python examples/train_dinov3_qwen.py --compile

# Increase data loading workers
python examples/train_dinov3_qwen.py --num-workers 8
```

## Next Steps

1. Replace dummy dataset with your actual image-text pairs
2. Adjust hyperparameters for your use case
3. Enable distributed training for multi-GPU setup
4. Add validation and evaluation metrics
