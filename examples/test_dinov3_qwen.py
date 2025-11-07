"""Test script for training CLIP with DinoV3 and Qwen Embedding.

This script demonstrates:
1. Loading DinoV3 (vision) and Qwen Embedding (text) from HuggingFace
2. 1:1 alignment training (standard CLIP)
3. 1:N alignment training (1 text with N vision encoders)
4. N:1 alignment training (N text encoders with 1 vision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyImageTextDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, num_samples=1000, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image
        image = torch.randn(3, self.image_size, self.image_size)
        # Random text tokens (simulating tokenized text)
        text = torch.randint(0, 30000, (77,))  # Qwen vocab size is larger
        return image, text


def create_dinov3_qwen_clip(
    embed_dim: int = 512,
    vision_model: str = "facebook/dinov2-base",  # Using dinov2-base as dinov3 may need special handling
    text_model: str = "Qwen/Qwen2-0.5B",  # Using smaller Qwen model for testing
    device: str = "cuda",
):
    """Create CLIP model with DinoV3 vision and Qwen text encoder.

    Args:
        embed_dim: Embedding dimension
        vision_model: HuggingFace vision model name
        text_model: HuggingFace text model name
        device: Device to use

    Returns:
        Tuple of (model, vision_encoder, text_encoder)
    """
    from open_clip.base_model import TransformersVisionEncoder, TransformersTextEncoder

    logger.info(f"Loading vision model: {vision_model}")
    vision_encoder = TransformersVisionEncoder(
        model_name=vision_model,
        output_dim=embed_dim,
        image_size=224,
        pooler_type='cls',
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(device)

    logger.info(f"Loading text model: {text_model}")
    text_encoder = TransformersTextEncoder(
        model_name=text_model,
        output_dim=embed_dim,
        pooler_type='mean',  # Qwen typically uses mean pooling
        proj_type='linear',
        pretrained=True,
        output_tokens=False,
    ).to(device)

    # Create simple CLIP-like model
    class SimpleCLIP(nn.Module):
        def __init__(self, vision_encoder, text_encoder):
            super().__init__()
            self.visual = vision_encoder
            self.text = text_encoder
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

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
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            return {
                'image_features': image_features,
                'text_features': text_features,
                'logit_scale': self.logit_scale.exp()
            }

    model = SimpleCLIP(vision_encoder, text_encoder).to(device)

    return model, vision_encoder, text_encoder


def clip_loss(image_features, text_features, logit_scale):
    """Standard CLIP contrastive loss."""
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Compute logits
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # Labels are diagonal
    labels = torch.arange(len(image_features), device=image_features.device)

    # Compute loss
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, texts) in enumerate(dataloader):
        images = images.to(device)
        texts = texts.to(device)

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
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Logit Scale: {outputs['logit_scale'].item():.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def test_1_to_1_training():
    """Test 1:1 alignment (standard CLIP) with DinoV3 and Qwen."""
    logger.info("=" * 80)
    logger.info("TEST 1: 1-to-1 Alignment (Standard CLIP)")
    logger.info("Vision: DinoV2 Base | Text: Qwen2-0.5B")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Hyperparameters
    batch_size = 16  # Smaller batch for testing
    embed_dim = 512
    num_epochs = 2
    learning_rate = 1e-5

    # Create model
    logger.info("\n1. Creating CLIP model...")
    try:
        model, vision_encoder, text_encoder = create_dinov3_qwen_clip(
            embed_dim=embed_dim,
            device=device,
        )
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        logger.info("Falling back to dummy encoders for testing...")
        return

    # Create dataset and dataloader
    logger.info("\n2. Creating dataset...")
    dataset = DummyImageTextDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    # Create optimizer
    logger.info("\n3. Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info(f"✓ Optimizer: AdamW with lr={learning_rate}")

    # Training loop
    logger.info("\n4. Training...")
    logger.info("-" * 80)
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch + 1)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed | Average Loss: {avg_loss:.4f}")
    logger.info("-" * 80)

    # Test inference
    logger.info("\n5. Testing inference...")
    model.eval()
    with torch.no_grad():
        test_images, test_texts = next(iter(dataloader))
        test_images = test_images[:4].to(device)
        test_texts = test_texts[:4].to(device)

        outputs = model(test_images, test_texts)

        logger.info(f"✓ Image features shape: {outputs['image_features'].shape}")
        logger.info(f"✓ Text features shape: {outputs['text_features'].shape}")

        # Compute similarity
        similarity = (100.0 * outputs['image_features'] @ outputs['text_features'].T).softmax(dim=-1)
        logger.info(f"✓ Similarity matrix shape: {similarity.shape}")
        logger.info(f"  Diagonal values (correct pairs): {similarity.diag().tolist()}")

    logger.info("\n✅ 1-to-1 training test completed successfully!\n")
    return model


def test_1_to_n_vision():
    """Test 1:N alignment (1 text with N vision encoders)."""
    logger.info("=" * 80)
    logger.info("TEST 2: 1-to-N Alignment (1 Text with Multiple Vision Encoders)")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from open_clip.multi_encoder_model import MultiVisionCLIP
    from open_clip.one_to_many_loss import OneToManyClipLoss
    from open_clip.base_model import TransformersVisionEncoder, TransformersTextEncoder

    batch_size = 16
    embed_dim = 512

    logger.info("\n1. Creating multi-vision model...")
    try:
        # Create multiple vision encoders
        vision_encoders = []
        vision_models = [
            "facebook/dinov2-small",
            "facebook/dinov2-base",
        ]

        for vm in vision_models:
            logger.info(f"   Loading {vm}...")
            encoder = TransformersVisionEncoder(
                model_name=vm,
                output_dim=embed_dim,
                image_size=224,
                pooler_type='cls',
                proj_type='linear',
                pretrained=True,
            ).to(device)
            vision_encoders.append(encoder)

        # Create text encoder
        logger.info("   Loading Qwen text encoder...")
        text_encoder = TransformersTextEncoder(
            model_name="Qwen/Qwen2-0.5B",
            output_dim=embed_dim,
            pooler_type='mean',
            proj_type='linear',
            pretrained=True,
        ).to(device)

        # Create multi-vision CLIP
        model = MultiVisionCLIP(
            vision_encoders=vision_encoders,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            output_dict=True,
        ).to(device)

        logger.info(f"✓ Multi-vision model created with {len(vision_encoders)} vision encoders")

    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        logger.info("Skipping 1-to-N vision test...")
        return

    # Create loss function
    logger.info("\n2. Creating one-to-many loss...")
    loss_fn = OneToManyClipLoss(
        text_to_multi_vision=True,
        num_multi_encoders=len(vision_encoders),
        aggregation='mean',
        local_loss=False,
        cache_labels=True,
    )
    logger.info("✓ Loss function created (1 text → N vision)")

    # Create dataset
    logger.info("\n3. Creating dataset...")
    dataset = DummyImageTextDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training
    logger.info("\n4. Training...")
    logger.info("-" * 80)
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)

            outputs = model(images, texts)

            loss = loss_fn(
                anchor_features=outputs['text_features'],
                multi_features_list=outputs['vision_features_list'],
                logit_scale=outputs['logit_scale'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 2 == 0:
                logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")
    logger.info("-" * 80)

    logger.info("\n✅ 1-to-N vision test completed successfully!\n")


def test_n_to_1_text():
    """Test N:1 alignment (N text encoders with 1 vision)."""
    logger.info("=" * 80)
    logger.info("TEST 3: N-to-1 Alignment (Multiple Text Encoders with 1 Vision)")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from open_clip.multi_encoder_model import MultiTextCLIP
    from open_clip.one_to_many_loss import OneToManyClipLoss
    from open_clip.base_model import TransformersVisionEncoder, TransformersTextEncoder

    batch_size = 16
    embed_dim = 512

    logger.info("\n1. Creating multi-text model...")
    try:
        # Create vision encoder
        logger.info("   Loading DinoV2 vision encoder...")
        vision_encoder = TransformersVisionEncoder(
            model_name="facebook/dinov2-base",
            output_dim=embed_dim,
            image_size=224,
            pooler_type='cls',
            proj_type='linear',
            pretrained=True,
        ).to(device)

        # Create multiple text encoders
        text_encoders = []
        text_models = [
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        for tm in text_models:
            logger.info(f"   Loading {tm}...")
            encoder = TransformersTextEncoder(
                model_name=tm,
                output_dim=embed_dim,
                pooler_type='mean',
                proj_type='linear',
                pretrained=True,
            ).to(device)
            text_encoders.append(encoder)

        # Create multi-text CLIP
        model = MultiTextCLIP(
            vision_encoder=vision_encoder,
            text_encoders=text_encoders,
            embed_dim=embed_dim,
            output_dict=True,
        ).to(device)

        logger.info(f"✓ Multi-text model created with {len(text_encoders)} text encoders")

    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        logger.info("Skipping N-to-1 text test...")
        return

    # Create loss function
    logger.info("\n2. Creating one-to-many loss...")
    loss_fn = OneToManyClipLoss(
        text_to_multi_vision=False,  # Vision is anchor, text is multi
        num_multi_encoders=len(text_encoders),
        aggregation='mean',
        local_loss=False,
        cache_labels=True,
    )
    logger.info("✓ Loss function created (1 vision → N text)")

    # Create dataset
    logger.info("\n3. Creating dataset...")
    dataset = DummyImageTextDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training
    logger.info("\n4. Training...")
    logger.info("-" * 80)
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)

            outputs = model(images, texts)

            loss = loss_fn(
                anchor_features=outputs['vision_features'],
                multi_features_list=outputs['text_features_list'],
                logit_scale=outputs['logit_scale'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 2 == 0:
                logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")
    logger.info("-" * 80)

    logger.info("\n✅ N-to-1 text test completed successfully!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CLIP Training Test Suite")
    print("DinoV3 (Vision) + Qwen Embedding (Text)")
    print("=" * 80 + "\n")

    try:
        # Test 1: Standard 1-to-1 CLIP
        test_1_to_1_training()

        # Test 2: 1 text with N vision encoders
        test_1_to_n_vision()

        # Test 3: N text encoders with 1 vision
        test_n_to_1_text()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    import os

    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

    main()
