"""Example: Multi-Encoder CLIP Training

This example demonstrates how to use the new multi-encoder features:
1. Multiple vision encoders with one text encoder
2. FSDP2 for efficient distributed training
3. One-to-many alignment loss
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import multi-encoder components
import sys
sys.path.insert(0, '../src')

from open_clip.multi_encoder_model import MultiVisionCLIP
from open_clip.one_to_many_loss import OneToManyClipLoss
from open_clip.timm_model import TimmModel
from open_clip.transformer import TextTransformer
from open_clip.base_model import TransformersVisionEncoder


def create_multi_vision_model(embed_dim=512):
    """Create a multi-vision CLIP model with 3 different vision encoders."""

    # Create multiple vision encoders with different architectures
    vision_encoders = [
        # ViT-B/16 at 224x224
        TimmModel(
            model_name='vit_base_patch16_224',
            embed_dim=embed_dim,
            image_size=224,
        ),
        # ConvNeXt Base
        TimmModel(
            model_name='convnext_base',
            embed_dim=embed_dim,
            image_size=224,
        ),
        # ResNet-50
        TimmModel(
            model_name='resnet50',
            embed_dim=embed_dim,
            image_size=224,
        ),
    ]

    # Create text encoder
    text_encoder = TextTransformer(
        context_length=77,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12,
    )

    # Create multi-vision CLIP
    model = MultiVisionCLIP(
        vision_encoders=vision_encoders,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        output_dict=True,
    )

    return model


def create_loss_function(world_size=1, rank=0):
    """Create one-to-many loss function."""
    return OneToManyClipLoss(
        text_to_multi_vision=True,  # One text aligns with multiple vision
        num_multi_encoders=3,
        aggregation='mean',  # Average features from all vision encoders
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=rank,
        world_size=world_size,
        use_horovod=False,
    )


def wrap_with_fsdp2(model):
    """Wrap model with FSDP2 for efficient distributed training."""
    try:
        from open_clip_train.fsdp2_utils import (
            is_fsdp2_available,
            wrap_model_with_fsdp2,
        )

        if not is_fsdp2_available():
            print("FSDP2 not available, using regular model")
            return model

        print("Wrapping model with FSDP2...")
        model = wrap_model_with_fsdp2(
            model,
            mp_policy='bf16',  # Use BF16 mixed precision
            reshard_after_forward=True,
        )
        print("FSDP2 wrapping complete")
        return model

    except Exception as e:
        print(f"FSDP2 wrapping failed: {e}")
        return model


def train_step(model, loss_fn, images, texts, optimizer):
    """Single training step."""
    # Forward pass
    outputs = model(images, texts)

    # Compute loss
    loss = loss_fn(
        anchor_features=outputs['text_features'],
        multi_features_list=outputs['vision_features_list'],
        logit_scale=outputs['logit_scale'],
    )

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def main():
    """Main training loop example."""
    print("=" * 80)
    print("Multi-Encoder CLIP Training Example")
    print("=" * 80)

    # Hyperparameters
    batch_size = 32
    embed_dim = 512
    num_epochs = 10
    learning_rate = 1e-4

    # Create model
    print("\n1. Creating multi-vision CLIP model...")
    model = create_multi_vision_model(embed_dim=embed_dim)
    print(f"   - Vision encoders: 3 (ViT-B/16, ConvNeXt, ResNet-50)")
    print(f"   - Text encoder: 1 (Transformer)")
    print(f"   - Embedding dimension: {embed_dim}")

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   - Device: {device}")

    # Optional: Wrap with FSDP2
    # model = wrap_with_fsdp2(model)

    # Create loss function
    print("\n2. Creating one-to-many loss function...")
    loss_fn = create_loss_function()
    print(f"   - Alignment: 1 text → 3 vision encoders")
    print(f"   - Aggregation: mean")

    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning rate: {learning_rate}")

    # Dummy data for demonstration
    print("\n4. Creating dummy data...")
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_texts = torch.randint(0, 49408, (batch_size, 77)).to(device)
    print(f"   - Images: {dummy_images.shape}")
    print(f"   - Texts: {dummy_texts.shape}")

    # Training loop
    print("\n5. Training loop...")
    print("-" * 80)
    for epoch in range(num_epochs):
        model.train()

        # Single training step with dummy data
        loss = train_step(model, loss_fn, dummy_images, dummy_texts, optimizer)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

    print("-" * 80)
    print("\n6. Training complete!")

    # Demonstrate inference
    print("\n7. Inference example...")
    model.eval()
    with torch.no_grad():
        # Encode images with all vision encoders
        vision_features_list = model.encode_image(dummy_images[:4])
        print(f"   - Vision features from 3 encoders:")
        for i, vf in enumerate(vision_features_list):
            print(f"     Encoder {i+1}: {vf.shape}")

        # Encode text
        text_features = model.encode_text(dummy_texts[:4])
        print(f"   - Text features: {text_features.shape}")

        # Compute similarities
        print("\n   - Computing similarities...")
        for i, vf in enumerate(vision_features_list):
            similarity = (100.0 * text_features @ vf.T).softmax(dim=-1)
            print(f"     Encoder {i+1} similarity: {similarity[0, 0].item():.4f}")

    print("\n" + "=" * 80)
    print("Example complete! See REFACTORING_GUIDE.md for more details.")
    print("=" * 80)


if __name__ == '__main__':
    main()
