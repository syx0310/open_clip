#!/usr/bin/env python3
"""
Complete training script for CLIP with DinoV3 and Qwen Embedding.

Supports three modes:
1. 1:1 alignment (standard CLIP)
2. 1:N alignment (1 text with N vision encoders)
3. N:1 alignment (N text encoders with 1 vision)

Usage:
    # Standard 1:1 training
    python train_dinov3_qwen.py

    # 1 text with 3 vision encoders
    python train_dinov3_qwen.py --mode 1text_nvision --num-vision 3

    # 2 text encoders with 1 vision
    python train_dinov3_qwen.py --mode ntext_1vision --num-text 2
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
from open_clip.one_to_many_loss import OneToManyClipLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset (Replace with your actual dataset)
# ============================================================================

class DummyImageTextDataset(Dataset):
    """Dummy dataset for testing. Replace with your actual dataset."""

    def __init__(self, num_samples, image_size, max_text_length):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_text_length = max_text_length
        logger.info(f"Created dummy dataset with {num_samples} samples")
        logger.info("⚠️  IMPORTANT: Replace this with your actual image-text dataset!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Replace with actual image and text loading
        # For real training, load actual images and tokenize text
        image = torch.randn(3, self.image_size, self.image_size)
        text = torch.randint(0, 50000, (self.max_text_length,))
        return image, text


# ============================================================================
# Model Creation Functions
# ============================================================================

def create_1to1_model(args):
    """Create standard 1:1 CLIP model."""
    logger.info("Creating 1:1 CLIP model (standard)")
    logger.info(f"  Vision: {args.vision_model}")
    logger.info(f"  Text: {args.text_model}")

    vision_encoder = TransformersVisionEncoder(
        model_name=args.vision_model,
        output_dim=args.embed_dim,
        image_size=args.image_size,
        pooler_type=args.vision_pooler,
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(args.device)

    text_encoder = TransformersTextEncoder(
        model_name=args.text_model,
        output_dim=args.embed_dim,
        pooler_type=args.text_pooler,
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(args.device)

    # Create simple CLIP wrapper
    class SimpleCLIP(nn.Module):
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

    model = SimpleCLIP(vision_encoder, text_encoder).to(args.device)
    logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_1text_nvision_model(args):
    """Create 1 text + N vision encoders model."""
    logger.info(f"Creating 1:N model (1 text with {args.num_vision} vision encoders)")

    # Create multiple vision encoders
    vision_encoders = []
    vision_models = [args.vision_model] * args.num_vision  # Same model, different instances

    for i, vm in enumerate(vision_models):
        logger.info(f"  Loading vision encoder {i+1}/{args.num_vision}: {vm}")
        encoder = TransformersVisionEncoder(
            model_name=vm,
            output_dim=args.embed_dim,
            image_size=args.image_size,
            pooler_type=args.vision_pooler,
            proj_type='linear',
            pretrained=True,
        ).to(args.device)
        vision_encoders.append(encoder)

    # Create text encoder
    logger.info(f"  Loading text encoder: {args.text_model}")
    text_encoder = TransformersTextEncoder(
        model_name=args.text_model,
        output_dim=args.embed_dim,
        pooler_type=args.text_pooler,
        proj_type='linear',
        pretrained=True,
    ).to(args.device)

    model = MultiVisionCLIP(
        vision_encoders=vision_encoders,
        text_encoder=text_encoder,
        embed_dim=args.embed_dim,
        output_dict=True,
    ).to(args.device)

    logger.info(f"✓ Multi-vision model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_ntext_1vision_model(args):
    """Create N text encoders + 1 vision model."""
    logger.info(f"Creating N:1 model ({args.num_text} text encoders with 1 vision)")

    # Create vision encoder
    logger.info(f"  Loading vision encoder: {args.vision_model}")
    vision_encoder = TransformersVisionEncoder(
        model_name=args.vision_model,
        output_dim=args.embed_dim,
        image_size=args.image_size,
        pooler_type=args.vision_pooler,
        proj_type='linear',
        pretrained=True,
    ).to(args.device)

    # Create multiple text encoders
    text_encoders = []
    text_models = [args.text_model] * args.num_text  # Same model, different instances

    for i, tm in enumerate(text_models):
        logger.info(f"  Loading text encoder {i+1}/{args.num_text}: {tm}")
        encoder = TransformersTextEncoder(
            model_name=tm,
            output_dim=args.embed_dim,
            pooler_type=args.text_pooler,
            proj_type='linear',
            pretrained=True,
        ).to(args.device)
        text_encoders.append(encoder)

    model = MultiTextCLIP(
        vision_encoder=vision_encoder,
        text_encoders=text_encoders,
        embed_dim=args.embed_dim,
        output_dict=True,
    ).to(args.device)

    logger.info(f"✓ Multi-text model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ============================================================================
# Loss Functions
# ============================================================================

def standard_clip_loss(image_features, text_features, logit_scale):
    """Standard CLIP contrastive loss."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    labels = torch.arange(len(image_features), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (images, texts) in enumerate(dataloader):
        images = images.to(args.device)
        texts = texts.to(args.device)

        # Forward pass
        outputs = model(images, texts)

        # Compute loss based on mode
        if args.mode == '1to1':
            loss = standard_clip_loss(
                outputs['image_features'],
                outputs['text_features'],
                outputs['logit_scale']
            )
        elif args.mode == '1text_nvision':
            loss = loss_fn(
                anchor_features=outputs['text_features'],
                multi_features_list=outputs['vision_features_list'],
                logit_scale=outputs['logit_scale'],
            )
        elif args.mode == 'ntext_1vision':
            loss = loss_fn(
                anchor_features=outputs['vision_features'],
                multi_features_list=outputs['text_features_list'],
                logit_scale=outputs['logit_scale'],
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % args.log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {loss.item():.4f} "
                f"Logit Scale: {outputs['logit_scale'].item():.2f} "
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
        'args': vars(args),
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CLIP with DinoV3 and Qwen')

    # Mode
    parser.add_argument('--mode', type=str, default='1to1',
                        choices=['1to1', '1text_nvision', 'ntext_1vision'],
                        help='Training mode')
    parser.add_argument('--num-vision', type=int, default=2,
                        help='Number of vision encoders (for 1text_nvision mode)')
    parser.add_argument('--num-text', type=int, default=2,
                        help='Number of text encoders (for ntext_1vision mode)')

    # Model
    parser.add_argument('--vision-model', type=str, default='facebook/dinov2-base',
                        help='Vision model from HuggingFace')
    parser.add_argument('--text-model', type=str, default='Qwen/Qwen2-0.5B',
                        help='Text model from HuggingFace')
    parser.add_argument('--embed-dim', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--vision-pooler', type=str, default='cls',
                        choices=['cls', 'mean', 'max'],
                        help='Vision pooling method')
    parser.add_argument('--text-pooler', type=str, default='mean',
                        choices=['cls', 'mean', 'max'],
                        help='Text pooling method')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Data
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of training samples (dummy data)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--max-text-length', type=int, default=77,
                        help='Max text length')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')

    # Loss aggregation for multi-encoder
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'weighted'],
                        help='Feature aggregation method for multi-encoder')

    args = parser.parse_args()

    # Setup device
    if not torch.cuda.is_available() and args.device == 'cuda':
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    logger.info("=" * 80)
    logger.info("CLIP Training: DinoV3 + Qwen Embedding")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Vision model: {args.vision_model}")
    logger.info(f"Text model: {args.text_model}")
    logger.info(f"Embedding dimension: {args.embed_dim}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 80)

    # Create model
    logger.info("\n[1/5] Creating model...")
    if args.mode == '1to1':
        model = create_1to1_model(args)
        loss_fn = None
    elif args.mode == '1text_nvision':
        model = create_1text_nvision_model(args)
        loss_fn = OneToManyClipLoss(
            text_to_multi_vision=True,
            num_multi_encoders=args.num_vision,
            aggregation=args.aggregation,
        )
    elif args.mode == 'ntext_1vision':
        model = create_ntext_1vision_model(args)
        loss_fn = OneToManyClipLoss(
            text_to_multi_vision=False,
            num_multi_encoders=args.num_text,
            aggregation=args.aggregation,
        )

    # Create dataset
    logger.info("\n[2/5] Creating dataset...")
    dataset = DummyImageTextDataset(
        num_samples=args.num_samples,
        image_size=args.image_size,
        max_text_length=args.max_text_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create optimizer
    logger.info("\n[3/5] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
    )
    logger.info(f"✓ Total training steps: {num_training_steps}")

    # Training
    logger.info("\n[4/5] Training...")
    logger.info("-" * 80)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, loss_fn, epoch, args
        )
        logger.info(f"✓ Epoch {epoch}/{args.epochs} completed | Avg Loss: {avg_loss:.4f}")

        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args)

    logger.info("-" * 80)

    # Final checkpoint
    logger.info("\n[5/5] Saving final model...")
    save_checkpoint(model, optimizer, args.epochs, avg_loss, args)

    # Test inference
    logger.info("\n[Bonus] Testing inference...")
    model.eval()
    with torch.no_grad():
        test_images, test_texts = next(iter(dataloader))
        test_images = test_images[:4].to(args.device)
        test_texts = test_texts[:4].to(args.device)

        outputs = model(test_images, test_texts)

        if args.mode == '1to1':
            img_feat = outputs['image_features']
            txt_feat = outputs['text_features']
            similarity = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            logger.info(f"✓ Similarity matrix (diagonal should be high):")
            logger.info(f"  {similarity.diag().tolist()}")
        elif args.mode == '1text_nvision':
            logger.info(f"✓ Text features: {outputs['text_features'].shape}")
            logger.info(f"✓ Vision features from {len(outputs['vision_features_list'])} encoders")
        elif args.mode == 'ntext_1vision':
            logger.info(f"✓ Vision features: {outputs['vision_features'].shape}")
            logger.info(f"✓ Text features from {len(outputs['text_features_list'])} encoders")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
