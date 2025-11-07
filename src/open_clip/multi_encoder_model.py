"""Multi-Encoder CLIP Models.

This module provides CLIP model variants that support:
1. Multiple vision encoders with a single text encoder
2. Multiple text encoders with a single vision encoder
3. Multiple encoders on both sides (M:N alignment)

This is useful for:
- Multi-view learning
- Ensemble models
- Cross-architecture knowledge distillation
"""
from typing import List, Optional, Union, Tuple, Dict
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model import CLIP, CLIPVisionCfg, CLIPTextCfg
from .base_model import BaseVisionEncoder, BaseTextEncoder


class MultiVisionCLIP(nn.Module):
    """CLIP with multiple vision encoders and one text encoder.

    This model supports 1:N alignment where one text embedding aligns
    with N vision embeddings from different vision encoders.

    Example:
        >>> vision_encoders = [
        ...     TimmModel('vit_base_patch16_224', 512),
        ...     TimmModel('convnext_base', 512),
        ... ]
        >>> text_encoder = TextTransformer(...)
        >>> model = MultiVisionCLIP(vision_encoders, text_encoder, embed_dim=512)
    """

    def __init__(
        self,
        vision_encoders: List[BaseVisionEncoder],
        text_encoder: BaseTextEncoder,
        embed_dim: int = 512,
        vision_projection: Optional[nn.Module] = None,
        text_projection: Optional[nn.Module] = None,
        logit_scale_init_value: float = 2.6592,  # ln(1/0.07)
        output_dict: bool = True,
    ):
        """Initialize MultiVisionCLIP.

        Args:
            vision_encoders: List of vision encoders
            text_encoder: Single text encoder
            embed_dim: Embedding dimension for projection
            vision_projection: Optional shared projection for vision features
            text_projection: Optional projection for text features
            logit_scale_init_value: Initial value for logit scale
            output_dict: Whether to return outputs as dict
        """
        super().__init__()
        self.vision_encoders = nn.ModuleList(vision_encoders)
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.output_dict = output_dict

        # Vision projections (one per encoder or shared)
        if vision_projection is not None:
            self.vision_projections = nn.ModuleList([vision_projection] * len(vision_encoders))
        else:
            self.vision_projections = nn.ModuleList([
                nn.Linear(enc.output_dim, embed_dim, bias=False)
                for enc in vision_encoders
            ])

        # Text projection
        if text_projection is not None:
            self.text_projection = text_projection
        else:
            self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim, bias=False)

        # Logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """Encode image with all vision encoders.

        Args:
            image: Input images (batch_size, C, H, W)
            normalize: Whether to L2-normalize the features

        Returns:
            List of vision features from each encoder
        """
        features_list = []
        for encoder, projection in zip(self.vision_encoders, self.vision_projections):
            features = encoder(image)
            if isinstance(features, tuple):
                features = features[0]  # Take pooled features if tokens also returned
            features = projection(features)
            if normalize:
                features = F.normalize(features, dim=-1)
            features_list.append(features)
        return features_list

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode text with text encoder.

        Args:
            text: Input text tokens (batch_size, seq_len)
            normalize: Whether to L2-normalize the features

        Returns:
            Text features
        """
        features = self.text_encoder(text)
        if isinstance(features, tuple):
            features = features[0]  # Take pooled features
        features = self.text_projection(features)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, List[torch.Tensor]], Dict]:
        """Forward pass.

        Args:
            image: Input images
            text: Input text tokens

        Returns:
            If output_dict=False: (text_features, vision_features_list)
            If output_dict=True: Dict with features and logit_scale
        """
        vision_features_list = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        if self.output_dict:
            return {
                "text_features": text_features,
                "vision_features_list": vision_features_list,
                "logit_scale": self.logit_scale.exp(),
            }
        else:
            return text_features, vision_features_list

    def lock_vision_tower(self, encoder_idx: Optional[int] = None, unlocked_groups: int = 0):
        """Lock vision encoder(s).

        Args:
            encoder_idx: Index of encoder to lock (None = lock all)
            unlocked_groups: Number of layer groups to leave unlocked
        """
        if encoder_idx is not None:
            self.vision_encoders[encoder_idx].lock(unlocked_groups=unlocked_groups)
        else:
            for encoder in self.vision_encoders:
                encoder.lock(unlocked_groups=unlocked_groups)

    def lock_text_tower(self, unlocked_layers: int = 0):
        """Lock text encoder.

        Args:
            unlocked_layers: Number of layers to leave unlocked
        """
        self.text_encoder.lock(unlocked_layers=unlocked_layers)


class MultiTextCLIP(nn.Module):
    """CLIP with multiple text encoders and one vision encoder.

    This model supports N:1 alignment where one vision embedding aligns
    with N text embeddings from different text encoders.
    """

    def __init__(
        self,
        vision_encoder: BaseVisionEncoder,
        text_encoders: List[BaseTextEncoder],
        embed_dim: int = 512,
        vision_projection: Optional[nn.Module] = None,
        text_projection: Optional[nn.Module] = None,
        logit_scale_init_value: float = 2.6592,
        output_dict: bool = True,
    ):
        """Initialize MultiTextCLIP.

        Args:
            vision_encoder: Single vision encoder
            text_encoders: List of text encoders
            embed_dim: Embedding dimension
            vision_projection: Optional projection for vision features
            text_projection: Optional shared projection for text features
            logit_scale_init_value: Initial value for logit scale
            output_dict: Whether to return outputs as dict
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoders = nn.ModuleList(text_encoders)
        self.embed_dim = embed_dim
        self.output_dict = output_dict

        # Vision projection
        if vision_projection is not None:
            self.vision_projection = vision_projection
        else:
            self.vision_projection = nn.Linear(vision_encoder.output_dim, embed_dim, bias=False)

        # Text projections (one per encoder or shared)
        if text_projection is not None:
            self.text_projections = nn.ModuleList([text_projection] * len(text_encoders))
        else:
            self.text_projections = nn.ModuleList([
                nn.Linear(enc.output_dim, embed_dim, bias=False)
                for enc in text_encoders
            ])

        # Logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode image.

        Args:
            image: Input images
            normalize: Whether to L2-normalize

        Returns:
            Vision features
        """
        features = self.vision_encoder(image)
        if isinstance(features, tuple):
            features = features[0]
        features = self.vision_projection(features)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """Encode text with all text encoders.

        Args:
            text: Input text tokens
            normalize: Whether to L2-normalize

        Returns:
            List of text features from each encoder
        """
        features_list = []
        for encoder, projection in zip(self.text_encoders, self.text_projections):
            features = encoder(text)
            if isinstance(features, tuple):
                features = features[0]
            features = projection(features)
            if normalize:
                features = F.normalize(features, dim=-1)
            features_list.append(features)
        return features_list

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], Dict]:
        """Forward pass.

        Args:
            image: Input images
            text: Input text tokens

        Returns:
            If output_dict=False: (text_features_list, vision_features)
            If output_dict=True: Dict
        """
        vision_features = self.encode_image(image, normalize=True)
        text_features_list = self.encode_text(text, normalize=True)

        if self.output_dict:
            return {
                "text_features_list": text_features_list,
                "vision_features": vision_features,
                "logit_scale": self.logit_scale.exp(),
            }
        else:
            return text_features_list, vision_features

    def lock_vision_tower(self, unlocked_groups: int = 0):
        """Lock vision encoder."""
        self.vision_encoder.lock(unlocked_groups=unlocked_groups)

    def lock_text_tower(self, encoder_idx: Optional[int] = None, unlocked_layers: int = 0):
        """Lock text encoder(s).

        Args:
            encoder_idx: Index of encoder to lock (None = lock all)
            unlocked_layers: Number of layers to leave unlocked
        """
        if encoder_idx is not None:
            self.text_encoders[encoder_idx].lock(unlocked_layers=unlocked_layers)
        else:
            for encoder in self.text_encoders:
                encoder.lock(unlocked_layers=unlocked_layers)


class MultiEncoderCLIP(nn.Module):
    """CLIP with multiple encoders on both sides (M:N alignment).

    This model supports full M:N alignment where M vision encoders
    align with N text encoders.
    """

    def __init__(
        self,
        vision_encoders: List[BaseVisionEncoder],
        text_encoders: List[BaseTextEncoder],
        embed_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        output_dict: bool = True,
    ):
        """Initialize MultiEncoderCLIP.

        Args:
            vision_encoders: List of vision encoders
            text_encoders: List of text encoders
            embed_dim: Embedding dimension
            logit_scale_init_value: Initial value for logit scale
            output_dict: Whether to return outputs as dict
        """
        super().__init__()
        self.vision_encoders = nn.ModuleList(vision_encoders)
        self.text_encoders = nn.ModuleList(text_encoders)
        self.embed_dim = embed_dim
        self.output_dict = output_dict

        # Projections
        self.vision_projections = nn.ModuleList([
            nn.Linear(enc.output_dim, embed_dim, bias=False)
            for enc in vision_encoders
        ])
        self.text_projections = nn.ModuleList([
            nn.Linear(enc.output_dim, embed_dim, bias=False)
            for enc in text_encoders
        ])

        # Logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """Encode image with all vision encoders."""
        features_list = []
        for encoder, projection in zip(self.vision_encoders, self.vision_projections):
            features = encoder(image)
            if isinstance(features, tuple):
                features = features[0]
            features = projection(features)
            if normalize:
                features = F.normalize(features, dim=-1)
            features_list.append(features)
        return features_list

    def encode_text(
        self,
        text: torch.Tensor,
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """Encode text with all text encoders."""
        features_list = []
        for encoder, projection in zip(self.text_encoders, self.text_projections):
            features = encoder(text)
            if isinstance(features, tuple):
                features = features[0]
            features = projection(features)
            if normalize:
                features = F.normalize(features, dim=-1)
            features_list.append(features)
        return features_list

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], Dict]:
        """Forward pass."""
        vision_features_list = self.encode_image(image, normalize=True)
        text_features_list = self.encode_text(text, normalize=True)

        if self.output_dict:
            return {
                "text_features_list": text_features_list,
                "vision_features_list": vision_features_list,
                "logit_scale": self.logit_scale.exp(),
            }
        else:
            return text_features_list, vision_features_list
