"""Base classes for custom text and vision encoders.

This module provides abstract base classes that standardize the interface for
text and vision encoders, making it easy to integrate custom models from
transformers or other sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple, List
import torch
import torch.nn as nn


class BaseVisionEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders.

    This class defines the standard interface that all vision encoders must implement.
    Custom vision encoders from transformers or other sources should inherit from this class.

    Attributes:
        output_dim (int): The dimension of the output embeddings
        image_size (Union[int, Tuple[int, int]]): Expected input image size
    """

    def __init__(self, output_dim: int, image_size: Union[int, Tuple[int, int]] = 224):
        """Initialize the base vision encoder.

        Args:
            output_dim: Dimension of output embeddings
            image_size: Expected input image size (square if int, or (height, width))
        """
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the vision encoder.

        Args:
            x: Input images of shape (batch_size, channels, height, width)

        Returns:
            If output_tokens is False:
                Tensor of shape (batch_size, output_dim) containing image embeddings
            If output_tokens is True:
                Tuple of (embeddings, tokens) where:
                    - embeddings: (batch_size, output_dim)
                    - tokens: (batch_size, num_tokens, token_dim)
        """
        pass

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        """Lock/freeze encoder parameters for transfer learning.

        Args:
            unlocked_groups: Number of layer groups to leave unlocked (0 = freeze all)
            freeze_bn_stats: Whether to freeze batch normalization statistics
        """
        if not unlocked_groups:
            for param in self.parameters():
                param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory efficiency.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        pass

    def get_num_layers(self) -> int:
        """Get the number of layers in the encoder.

        Returns:
            Number of layers
        """
        return 0


class BaseTextEncoder(ABC, nn.Module):
    """Abstract base class for text encoders.

    This class defines the standard interface that all text encoders must implement.
    Custom text encoders from transformers or other sources should inherit from this class.

    Attributes:
        output_dim (int): The dimension of the output embeddings
        vocab_size (int): Size of the vocabulary
        context_length (int): Maximum sequence length
    """

    def __init__(self, output_dim: int, vocab_size: int = 0, context_length: int = 77):
        """Initialize the base text encoder.

        Args:
            output_dim: Dimension of output embeddings
            vocab_size: Size of vocabulary (0 if using external tokenizer)
            context_length: Maximum sequence length
        """
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.context_length = context_length

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the text encoder.

        Args:
            x: Input token IDs of shape (batch_size, sequence_length)

        Returns:
            If output_tokens is False:
                Tensor of shape (batch_size, output_dim) containing text embeddings
            If output_tokens is True:
                Tuple of (embeddings, tokens) where:
                    - embeddings: (batch_size, output_dim)
                    - tokens: (batch_size, num_tokens, token_dim)
        """
        pass

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """Lock/freeze encoder parameters for transfer learning.

        Args:
            unlocked_layers: Number of layers to leave unlocked (0 = freeze all)
            freeze_layer_norm: Whether to freeze LayerNorm parameters
        """
        if not unlocked_layers:
            for param in self.parameters():
                param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory efficiency.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        pass

    def get_num_layers(self) -> int:
        """Get the number of layers in the encoder.

        Returns:
            Number of layers
        """
        return 0


class TransformersVisionEncoder(BaseVisionEncoder):
    """Vision encoder adapter for HuggingFace transformers models.

    This class wraps any HuggingFace vision transformer model (ViT, DINO, etc.)
    to work with the OpenCLIP framework.

    Example:
        >>> from transformers import ViTModel
        >>> encoder = TransformersVisionEncoder(
        ...     model_name="google/vit-base-patch16-224",
        ...     output_dim=512,
        ...     pretrained=True
        ... )
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        image_size: Union[int, Tuple[int, int]] = 224,
        pooler_type: str = 'cls',
        proj_type: str = 'linear',
        pretrained: bool = True,
        output_tokens: bool = False,
    ):
        """Initialize transformers vision encoder.

        Args:
            model_name: HuggingFace model name or path
            output_dim: Dimension of output embeddings
            image_size: Expected input image size
            pooler_type: Pooling strategy ('cls', 'mean', 'max')
            proj_type: Projection type ('linear', 'mlp', 'none')
            pretrained: Whether to load pretrained weights
            output_tokens: Whether to return all tokens in addition to pooled output
        """
        super().__init__(output_dim, image_size)
        self.output_tokens = output_tokens

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise RuntimeError("Please `pip install transformers` to use transformers-based vision models")

        # Load model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
            self.config = self.model.config
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(self.config)

        # Get hidden size
        hidden_size = self.config.hidden_size

        # Setup pooling
        self.pooler_type = pooler_type

        # Setup projection
        if proj_type == 'linear':
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_proj = (hidden_size + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_proj, bias=False),
                nn.GELU(),
                nn.Linear(hidden_proj, output_dim, bias=False),
            )
        elif proj_type == 'none' or (hidden_size == output_dim):
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the vision encoder.

        Args:
            x: Input images (batch_size, channels, height, width)

        Returns:
            embeddings or (embeddings, tokens)
        """
        outputs = self.model(pixel_values=x, return_dict=True)

        # Pool features
        if self.pooler_type == 'cls':
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooler_type == 'mean':
            pooled = outputs.last_hidden_state.mean(dim=1)
        elif self.pooler_type == 'max':
            pooled = outputs.last_hidden_state.max(dim=1).values
        else:
            pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]

        # Project
        projected = self.proj(pooled)

        if self.output_tokens:
            tokens = outputs.last_hidden_state[:, 1:]  # Exclude CLS token
            return projected, tokens
        return projected

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        """Lock encoder parameters."""
        if not unlocked_groups:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Unlock last N layers
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                layers = self.model.encoder.layer
                for layer in layers[:-unlocked_groups]:
                    for param in layer.parameters():
                        param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable() if enable else self.model.gradient_checkpointing_disable()


class TransformersTextEncoder(BaseTextEncoder):
    """Text encoder adapter for HuggingFace transformers models.

    This class wraps any HuggingFace text model (BERT, RoBERTa, GPT, etc.)
    to work with the OpenCLIP framework.

    Example:
        >>> from transformers import BertModel
        >>> encoder = TransformersTextEncoder(
        ...     model_name="bert-base-uncased",
        ...     output_dim=512,
        ...     pretrained=True
        ... )
    """

    def __init__(
        self,
        model_name: str,
        output_dim: int,
        pooler_type: str = 'cls',
        proj_type: str = 'linear',
        pretrained: bool = True,
        output_tokens: bool = False,
    ):
        """Initialize transformers text encoder.

        Args:
            model_name: HuggingFace model name or path
            output_dim: Dimension of output embeddings
            pooler_type: Pooling strategy ('cls', 'mean', 'max')
            proj_type: Projection type ('linear', 'mlp', 'none')
            pretrained: Whether to load pretrained weights
            output_tokens: Whether to return all tokens in addition to pooled output
        """
        try:
            from transformers import AutoModel, AutoConfig, AutoTokenizer
        except ImportError:
            raise RuntimeError("Please `pip install transformers` to use transformers-based text models")

        # Load config first to get vocab_size and max_length
        self.config = AutoConfig.from_pretrained(model_name)
        vocab_size = getattr(self.config, 'vocab_size', 0)
        context_length = getattr(self.config, 'max_position_embeddings', 77)

        super().__init__(output_dim, vocab_size, context_length)
        self.output_tokens = output_tokens

        # Load model
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(self.config)

        # Get hidden size
        hidden_size = self.config.hidden_size

        # Setup pooling
        self.pooler_type = pooler_type

        # Setup projection
        if proj_type == 'linear':
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_proj = (hidden_size + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_proj, bias=False),
                nn.GELU(),
                nn.Linear(hidden_proj, output_dim, bias=False),
            )
        elif proj_type == 'none' or (hidden_size == output_dim):
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the text encoder.

        Args:
            x: Input token IDs (batch_size, sequence_length)

        Returns:
            embeddings or (embeddings, tokens)
        """
        # Create attention mask
        attention_mask = (x != 0).long()  # Assume 0 is padding token

        outputs = self.model(input_ids=x, attention_mask=attention_mask, return_dict=True)

        # Pool features
        if self.pooler_type == 'cls':
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooler_type == 'mean':
            # Mean pooling with attention mask
            masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif self.pooler_type == 'max':
            masked_output = outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), -torch.inf)
            pooled = masked_output.max(dim=1).values
        else:
            pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]

        # Project
        projected = self.proj(pooled)

        if self.output_tokens:
            tokens = outputs.last_hidden_state
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        """Lock encoder parameters."""
        if not unlocked_layers:
            for n, p in self.model.named_parameters():
                if freeze_layer_norm:
                    p.requires_grad = False
                else:
                    p.requires_grad = "LayerNorm" in n
        else:
            # Unlock last N layers
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                layers = self.model.encoder.layer
                for layer in layers[:-unlocked_layers]:
                    for n, p in layer.named_parameters():
                        if freeze_layer_norm:
                            p.requires_grad = False
                        else:
                            p.requires_grad = "LayerNorm" in n

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable() if enable else self.model.gradient_checkpointing_disable()
