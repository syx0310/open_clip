"""Simple training script for DinoV3 (vision) + Qwen Embedding (text).

This is a minimal, production-ready script for training CLIP with:
- Vision: DinoV3 (facebook/dinov2-base or dinov2-large)
- Text: Qwen Embedding (Qwen/Qwen2-0.5B or larger)

Usage:
    python simple_dinov3_qwen_training.py

Requirements:
    pip install torch transformers pillow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration."""
    # Model
    vision_model = "facebook/dinov2-base"  # or "facebook/dinov2-large"
    text_model = "Qwen/Qwen2-0.5B"  # or "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B"
    embed_dim = 768  # Embedding dimension

    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    warmup_steps = 500
    weight_decay = 0.01

    # Data
    num_train_samples = 1000  # For dummy data
    image_size = 224
    max_text_length = 77

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    save_dir = Path("./checkpoints")
    save_every = 5  # Save checkpoint every N epochs


# ============================================================================
# Dataset (Replace with your actual dataset)
# ============================================================================

class DummyImageTextDataset(Dataset):
    """Dummy dataset - replace with your actual data loading."""

    def __init__(self, num_samples, image_size, max_text_length):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_text_length = max_text_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Replace with actual image and text loading
        # Image should be normalized to [0, 1] or ImageNet stats
        image = torch.randn(3, self.image_size, self.image_size)

        # Text tokens - replace with actual tokenizer output
        text = torch.randint(0, 30000, (self.max_text_length,))

        return image, text


# ============================================================================
# Model
# ============================================================================

class DinoV3QwenCLIP(nn.Module):
    """CLIP model with DinoV3 vision and Qwen text encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load vision encoder (DinoV3)
        logger.info(f"Loading vision encoder: {config.vision_model}")
        from transformers import AutoModel, AutoImageProcessor
        self.vision_model = AutoModel.from_pretrained(config.vision_model)
        self.image_processor = AutoImageProcessor.from_pretrained(config.vision_model)

        # Get vision hidden size
        vision_hidden_size = self.vision_model.config.hidden_size

        # Load text encoder (Qwen)
        logger.info(f"Loading text encoder: {config.text_model}")
        from transformers import AutoModel as AutoTextModel, AutoTokenizer
        self.text_model = AutoTextModel.from_pretrained(
            config.text_model,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_model,
            trust_remote_code=True,
        )

        # Get text hidden size
        text_hidden_size = self.text_model.config.hidden_size

        # Projection layers
        self.vision_projection = nn.Linear(vision_hidden_size, config.embed_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, config.embed_dim, bias=False)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

        logger.info(f"Vision hidden size: {vision_hidden_size}")
        logger.info(f"Text hidden size: {text_hidden_size}")
        logger.info(f"Projection dimension: {config.embed_dim}")

    def encode_image(self, pixel_values, normalize=True):
        """Encode images to embeddings.

        Args:
            pixel_values: Image tensors (batch_size, 3, 224, 224)
            normalize: Whether to L2-normalize embeddings

        Returns:
            Image embeddings (batch_size, embed_dim)
        """
        # Forward through vision model
        outputs = self.vision_model(pixel_values=pixel_values)

        # Get CLS token (first token)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Project to common embedding space
        embeddings = self.vision_projection(pooled_output)

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def encode_text(self, input_ids, attention_mask=None, normalize=True):
        """Encode text to embeddings.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            normalize: Whether to L2-normalize embeddings

        Returns:
            Text embeddings (batch_size, embed_dim)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        # Forward through text model
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        pooled_output = sum_hidden / sum_mask

        # Project to common embedding space
        embeddings = self.text_projection(pooled_output)

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, images, texts):
        """Forward pass.

        Args:
            images: Image tensors
            texts: Text token IDs

        Returns:
            Dict with image_features, text_features, and logit_scale
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        return {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp()
        }


# ============================================================================
# Loss Function
# ============================================================================

def clip_loss(image_features, text_features, logit_scale):
    """Compute CLIP contrastive loss.

    Args:
        image_features: Normalized image embeddings (batch_size, embed_dim)
        text_features: Normalized text embeddings (batch_size, embed_dim)
        logit_scale: Temperature parameter

    Returns:
        Loss value
    """
    # Compute similarity matrix
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # Ground truth: diagonal is positive pairs
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    # Symmetric loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    loss = (loss_i2t + loss_t2i) / 2

    return loss


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (images, texts) in enumerate(dataloader):
        images = images.to(config.device)
        texts = texts.to(config.device)

        # Forward pass
        outputs = model(images, texts)

        # Compute loss
        loss = clip_loss(
            outputs['image_features'],
            outputs['text_features'],
            outputs['logit_scale']
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {loss.item():.4f} "
                f"Logit Scale: {outputs['logit_scale'].item():.2f} "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, config):
    """Save model checkpoint."""
    config.save_dir.mkdir(exist_ok=True, parents=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': vars(config),
    }

    checkpoint_path = config.save_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def evaluate(model, dataloader, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(config.device)
            texts = texts.to(config.device)

            outputs = model(images, texts)
            loss = clip_loss(
                outputs['image_features'],
                outputs['text_features'],
                outputs['logit_scale']
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training function."""
    config = Config()

    logger.info("=" * 80)
    logger.info("CLIP Training: DinoV3 + Qwen Embedding")
    logger.info("=" * 80)
    logger.info(f"Vision model: {config.vision_model}")
    logger.info(f"Text model: {config.text_model}")
    logger.info(f"Embedding dimension: {config.embed_dim}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("=" * 80)

    # Create model
    logger.info("\n[1/5] Creating model...")
    model = DinoV3QwenCLIP(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Model created with {num_params:,} trainable parameters")

    # Create dataset and dataloader
    logger.info("\n[2/5] Creating dataset...")
    train_dataset = DummyImageTextDataset(
        num_samples=config.num_train_samples,
        image_size=config.image_size,
        max_text_length=config.max_text_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"✓ Dataset created with {len(train_dataset)} samples")

    # Create optimizer and scheduler
    logger.info("\n[3/5] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
    )
    logger.info(f"✓ Optimizer and scheduler created")
    logger.info(f"  Total training steps: {num_training_steps}")

    # Training loop
    logger.info("\n[4/5] Training...")
    logger.info("-" * 80)

    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, config)
        logger.info(f"Epoch {epoch} - Average training loss: {train_loss:.4f}")

        # Save checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, config)

    logger.info("-" * 80)

    # Final evaluation
    logger.info("\n[5/5] Final evaluation...")
    eval_loss = evaluate(model, train_loader, config)
    logger.info(f"✓ Final evaluation loss: {eval_loss:.4f}")

    # Save final model
    save_checkpoint(model, optimizer, config.num_epochs, eval_loss, config)

    # Test inference
    logger.info("\n[Bonus] Testing inference...")
    model.eval()
    with torch.no_grad():
        test_images, test_texts = next(iter(train_loader))
        test_images = test_images[:4].to(config.device)
        test_texts = test_texts[:4].to(config.device)

        outputs = model(test_images, test_texts)

        # Compute similarity
        similarity = (100.0 * outputs['image_features'] @ outputs['text_features'].T)
        similarity = similarity.softmax(dim=-1)

        logger.info(f"✓ Image features: {outputs['image_features'].shape}")
        logger.info(f"✓ Text features: {outputs['text_features'].shape}")
        logger.info(f"✓ Similarity matrix:\n{similarity}")
        logger.info(f"✓ Correct pair similarities (diagonal): {similarity.diag().tolist()}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
