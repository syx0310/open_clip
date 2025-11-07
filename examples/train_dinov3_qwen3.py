#!/usr/bin/env python3
"""
Training script for DinoV3 (vision) + Qwen3-Embedding (text) CLIP model.

Models:
- Vision: facebook/dinov3-vith16plus-pretrain-lvd1689m (840M params)
- Text: Qwen/Qwen3-Embedding-4B (4B params)

Usage:
    # Standard 1:1 training
    python train_dinov3_qwen3.py

    # With custom arguments
    python train_dinov3_qwen3.py --batch-size 8 --embed-dim 1024
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_clip.base_model import TransformersVisionEncoder, TransformersTextEncoder
from open_clip.multi_encoder_model import MultiVisionCLIP, MultiTextCLIP

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class DummyImageTextDataset(Dataset):
    """Dummy dataset. Replace with your actual image-text dataset."""

    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size
        logger.warning("⚠️  Using dummy dataset! Replace with your actual data.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Load actual images and tokenize text
        # For DinoV3, images should be normalized
        image = torch.randn(3, self.image_size, self.image_size)

        # For Qwen3-Embedding, use proper tokenization
        # The model supports up to 32K context length
        text = torch.randint(0, 50000, (512,))  # Qwen3 vocab size is ~150K

        return image, text


# ============================================================================
# Model
# ============================================================================

def create_dinov3_qwen3_clip(args):
    """Create CLIP model with DinoV3 vision and Qwen3-Embedding text."""
    logger.info("=" * 80)
    logger.info("Creating DinoV3 + Qwen3-Embedding CLIP Model")
    logger.info("=" * 80)

    # Create vision encoder (DinoV3)
    logger.info(f"Loading DinoV3 vision encoder: {args.vision_model}")
    logger.info(f"  Architecture: ViT-H+ with 840M parameters")
    logger.info(f"  Input size: {args.image_size}x{args.image_size}")

    vision_encoder = TransformersVisionEncoder(
        model_name=args.vision_model,
        output_dim=args.embed_dim,
        image_size=args.image_size,
        pooler_type='cls',  # Use CLS token for DinoV3
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(args.device)

    logger.info("✓ DinoV3 vision encoder loaded")

    # Create text encoder (Qwen3-Embedding)
    logger.info(f"Loading Qwen3-Embedding text encoder: {args.text_model}")
    logger.info(f"  Architecture: 4B parameters, 32K context")
    logger.info(f"  Max embedding dim: 2560")

    text_encoder = TransformersTextEncoder(
        model_name=args.text_model,
        output_dim=args.embed_dim,
        pooler_type='mean',  # Mean pooling for embeddings
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(args.device)

    logger.info("✓ Qwen3-Embedding text encoder loaded")

    # Create CLIP wrapper
    class DinoV3Qwen3CLIP(nn.Module):
        def __init__(self, vision_encoder, text_encoder):
            super().__init__()
            self.visual = vision_encoder
            self.text = text_encoder
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        def encode_image(self, image, normalize=True):
            features = self.visual(image)
            if normalize:
                features = F.normalize(features, dim=-1)
            return features

        def encode_text(self, text, normalize=True):
            features = self.text(text)
            if normalize:
                features = F.normalize(features, dim=-1)
            return features

        def forward(self, image, text):
            return {
                'image_features': self.encode_image(image),
                'text_features': self.encode_text(text),
                'logit_scale': self.logit_scale.exp()
            }

    model = DinoV3Qwen3CLIP(vision_encoder, text_encoder).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 80)
    logger.info(f"✓ Model created successfully")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Embedding dimension: {args.embed_dim}")
    logger.info("=" * 80)

    return model


# ============================================================================
# Loss & Training
# ============================================================================

def clip_loss(image_features, text_features, logit_scale):
    """CLIP contrastive loss."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    labels = torch.arange(len(image_features), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


def train_epoch(model, dataloader, optimizer, scheduler, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (images, texts) in enumerate(dataloader):
        images = images.to(args.device)
        texts = texts.to(args.device)

        # Forward
        outputs = model(images, texts)
        loss = clip_loss(
            outputs['image_features'],
            outputs['text_features'],
            outputs['logit_scale']
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            logger.info(
                f"Epoch {epoch:2d} [{batch_idx:3d}/{num_batches:3d}] "
                f"Loss: {loss.item():.4f} "
                f"Scale: {outputs['logit_scale'].item():.2f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, args):
    """Save checkpoint."""
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': vars(args),
    }

    path = checkpoint_dir / f"dinov3_qwen3_epoch_{epoch}.pt"
    torch.save(checkpoint, path)
    logger.info(f"✓ Checkpoint saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CLIP with DinoV3 and Qwen3-Embedding'
    )

    # Models
    parser.add_argument(
        '--vision-model',
        type=str,
        default='facebook/dinov3-vith16plus-pretrain-lvd1689m',
        help='DinoV3 vision model from HuggingFace'
    )
    parser.add_argument(
        '--text-model',
        type=str,
        default='Qwen/Qwen3-Embedding-4B',
        help='Qwen3-Embedding text model from HuggingFace'
    )
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=1024,
        help='Embedding dimension (max 2560 for Qwen3)'
    )

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Warmup steps')

    # Data
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples (dummy data)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (DinoV3 uses 224)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='./checkpoints_dinov3_qwen3',
                        help='Checkpoint directory')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save interval (epochs)')

    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("DinoV3 + Qwen3-Embedding CLIP Training")
    logger.info("=" * 80)
    logger.info(f"Vision: {args.vision_model}")
    logger.info(f"Text: {args.text_model}")
    logger.info(f"Embed dim: {args.embed_dim}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 80 + "\n")

    # Create model
    logger.info("[1/5] Creating model...")
    model = create_dinov3_qwen3_clip(args)

    # Create dataset
    logger.info("\n[2/5] Creating dataset...")
    dataset = DummyImageTextDataset(args.num_samples, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"✓ Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Create optimizer
    logger.info("\n[3/5] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    logger.info(f"✓ Optimizer: AdamW, Total steps: {total_steps}")

    # Training
    logger.info("\n[4/5] Training...")
    logger.info("-" * 80)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, epoch, args)
        logger.info(f"✓ Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args)

    logger.info("-" * 80)

    # Save final model
    logger.info("\n[5/5] Saving final model...")
    save_checkpoint(model, optimizer, args.epochs, avg_loss, args)

    # Test
    logger.info("\n[Test] Running inference test...")
    model.eval()
    with torch.no_grad():
        test_images, test_texts = next(iter(dataloader))
        test_images = test_images[:4].to(args.device)
        test_texts = test_texts[:4].to(args.device)

        outputs = model(test_images, test_texts)

        similarity = (100.0 * outputs['image_features'] @ outputs['text_features'].T)
        similarity = similarity.softmax(dim=-1)

        logger.info(f"✓ Image features: {outputs['image_features'].shape}")
        logger.info(f"✓ Text features: {outputs['text_features'].shape}")
        logger.info(f"✓ Similarity matrix (diagonal):")
        for i, sim in enumerate(similarity.diag()):
            logger.info(f"    Pair {i}: {sim.item():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info(f"Checkpoints: {args.checkpoint_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
