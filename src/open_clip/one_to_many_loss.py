"""One-to-Many Alignment Loss for CLIP.

This module implements loss functions for one-to-many alignment scenarios:
1. One text embedding aligned with multiple vision embeddings
2. One vision embedding aligned with multiple text embeddings

This is useful for:
- Multi-view learning (one text, multiple image views)
- Multi-caption learning (one image, multiple captions)
- Ensemble models (one modality with multiple encoders for the other)
"""
from typing import Optional, List, Union
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features_one_to_many(
    anchor_features,
    multi_features_list,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False
):
    """Gather features for one-to-many alignment across GPUs.

    Args:
        anchor_features: Single set of features (batch_size, embed_dim)
        multi_features_list: List of feature sets, each (batch_size, embed_dim)
        local_loss: Whether to compute loss locally per GPU
        gather_with_grad: Whether to gather with gradient
        rank: Current process rank
        world_size: Total number of processes
        use_horovod: Whether to use Horovod for gathering

    Returns:
        Tuple of (gathered_anchor_features, gathered_multi_features_list)
    """
    assert has_distributed, 'torch.distributed did not import correctly'

    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_anchor_features = hvd.allgather(anchor_features)
            all_multi_features_list = [hvd.allgather(mf) for mf in multi_features_list]
        else:
            with torch.no_grad():
                all_anchor_features = hvd.allgather(anchor_features)
                all_multi_features_list = [hvd.allgather(mf) for mf in multi_features_list]
            if not local_loss:
                # Ensure grads for local rank
                gathered_anchor = list(all_anchor_features.chunk(world_size, dim=0))
                gathered_anchor[rank] = anchor_features
                all_anchor_features = torch.cat(gathered_anchor, dim=0)

                all_multi_features_list = [
                    torch.cat([
                        mf if i == rank else chunk
                        for i, chunk in enumerate(all_mf.chunk(world_size, dim=0))
                    ], dim=0)
                    for mf, all_mf in zip(multi_features_list, all_multi_features_list)
                ]
    else:
        # PyTorch distributed gathering
        if gather_with_grad:
            all_anchor_features = torch.cat(
                torch.distributed.nn.all_gather(anchor_features), dim=0
            )
            all_multi_features_list = [
                torch.cat(torch.distributed.nn.all_gather(mf), dim=0)
                for mf in multi_features_list
            ]
        else:
            gathered_anchor = [torch.zeros_like(anchor_features) for _ in range(world_size)]
            dist.all_gather(gathered_anchor, anchor_features)
            if not local_loss:
                gathered_anchor[rank] = anchor_features
            all_anchor_features = torch.cat(gathered_anchor, dim=0)

            all_multi_features_list = []
            for mf in multi_features_list:
                gathered_mf = [torch.zeros_like(mf) for _ in range(world_size)]
                dist.all_gather(gathered_mf, mf)
                if not local_loss:
                    gathered_mf[rank] = mf
                all_multi_features_list.append(torch.cat(gathered_mf, dim=0))

    return all_anchor_features, all_multi_features_list


class OneToManyClipLoss(nn.Module):
    """One-to-Many CLIP Loss.

    This loss supports scenarios where:
    1. One text aligns with N vision encoders (text_to_multi_vision=True)
    2. One vision aligns with N text encoders (text_to_multi_vision=False)

    The loss is computed by:
    1. Computing similarities between anchor and each multi-encoder output
    2. Aggregating similarities (mean, max, or weighted sum)
    3. Computing contrastive loss on aggregated similarities

    Example:
        >>> # One text with 3 vision encoders
        >>> loss_fn = OneToManyClipLoss(
        ...     text_to_multi_vision=True,
        ...     num_multi_encoders=3,
        ...     aggregation='mean'
        ... )
        >>> text_features = torch.randn(32, 512)
        >>> vision_features_list = [
        ...     torch.randn(32, 512),
        ...     torch.randn(32, 512),
        ...     torch.randn(32, 512)
        ... ]
        >>> loss = loss_fn(text_features, vision_features_list, logit_scale=1.0)
    """

    def __init__(
        self,
        text_to_multi_vision: bool = True,
        num_multi_encoders: int = 1,
        aggregation: str = 'mean',
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
        learnable_weights: bool = False,
    ):
        """Initialize OneToManyClipLoss.

        Args:
            text_to_multi_vision: If True, one text aligns with N visions.
                If False, one vision aligns with N texts.
            num_multi_encoders: Number of encoders for the multi-modality side
            aggregation: How to aggregate multiple features ('mean', 'max', 'weighted')
            local_loss: Compute loss locally per GPU
            gather_with_grad: Gather features with gradient
            cache_labels: Cache ground truth labels
            rank: Current process rank
            world_size: Total number of processes
            use_horovod: Use Horovod for distributed training
            learnable_weights: Learn weights for weighted aggregation
        """
        super().__init__()
        self.text_to_multi_vision = text_to_multi_vision
        self.num_multi_encoders = num_multi_encoders
        self.aggregation = aggregation
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # Learnable weights for aggregation
        if learnable_weights and aggregation == 'weighted':
            self.aggregation_weights = nn.Parameter(
                torch.ones(num_multi_encoders) / num_multi_encoders
            )
        else:
            self.aggregation_weights = None

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """Get ground truth labels for contrastive loss."""
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def aggregate_features(
        self,
        multi_features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate multiple feature sets into one.

        Args:
            multi_features_list: List of feature tensors, each (batch_size, embed_dim)

        Returns:
            Aggregated features (batch_size, embed_dim)
        """
        if len(multi_features_list) == 1:
            return multi_features_list[0]

        # Stack all features
        stacked = torch.stack(multi_features_list, dim=1)  # (batch, N, embed_dim)

        if self.aggregation == 'mean':
            return stacked.mean(dim=1)
        elif self.aggregation == 'max':
            return stacked.max(dim=1).values
        elif self.aggregation == 'weighted':
            if self.aggregation_weights is not None:
                # Normalize weights
                weights = F.softmax(self.aggregation_weights, dim=0)
                weights = weights.view(1, -1, 1)  # (1, N, 1)
                return (stacked * weights).sum(dim=1)
            else:
                # Default to mean if weights not initialized
                return stacked.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def get_logits(
        self,
        anchor_features: torch.Tensor,
        multi_features_list: List[torch.Tensor],
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
    ):
        """Compute logits for one-to-many alignment.

        Args:
            anchor_features: Features from single encoder (batch_size, embed_dim)
            multi_features_list: List of features from N encoders
            logit_scale: Scale factor for logits
            logit_bias: Optional bias for logits

        Returns:
            Tuple of (logits_per_anchor, logits_per_multi)
        """
        # Gather features across GPUs if distributed
        if self.world_size > 1:
            all_anchor_features, all_multi_features_list = gather_features_one_to_many(
                anchor_features,
                multi_features_list,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
        else:
            all_anchor_features = anchor_features
            all_multi_features_list = multi_features_list

        # Aggregate multiple features
        aggregated_multi_features = self.aggregate_features(all_multi_features_list)

        # Compute logits
        if self.world_size > 1 and self.local_loss:
            # Local loss: use local anchor with global multi
            logits_per_anchor = logit_scale * anchor_features @ aggregated_multi_features.T
            logits_per_multi = logit_scale * aggregated_multi_features @ anchor_features.T
        else:
            # Global loss: use gathered features
            logits_per_anchor = logit_scale * all_anchor_features @ aggregated_multi_features.T
            logits_per_multi = logits_per_anchor.T

        if logit_bias is not None:
            logits_per_anchor = logits_per_anchor + logit_bias
            logits_per_multi = logits_per_multi + logit_bias

        return logits_per_anchor, logits_per_multi

    def forward(
        self,
        anchor_features: torch.Tensor,
        multi_features_list: Union[List[torch.Tensor], torch.Tensor],
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        """Forward pass for one-to-many loss.

        Args:
            anchor_features: Features from single encoder (batch_size, embed_dim)
            multi_features_list: List of features from N encoders, or single tensor
            logit_scale: Scale factor for logits
            logit_bias: Optional bias for logits
            output_dict: Whether to return dict with detailed outputs

        Returns:
            Loss value or dict with loss and additional info
        """
        # Handle case where multi_features_list is a single tensor
        if isinstance(multi_features_list, torch.Tensor):
            multi_features_list = [multi_features_list]

        device = anchor_features.device

        # Compute logits
        logits_per_anchor, logits_per_multi = self.get_logits(
            anchor_features,
            multi_features_list,
            logit_scale,
            logit_bias=logit_bias,
        )

        # Get ground truth labels
        labels = self.get_ground_truth(device, logits_per_anchor.shape[0])

        # Compute contrastive loss
        if self.text_to_multi_vision:
            # anchor = text, multi = vision
            loss_text = F.cross_entropy(logits_per_anchor, labels)
            loss_vision = F.cross_entropy(logits_per_multi, labels)
        else:
            # anchor = vision, multi = text
            loss_vision = F.cross_entropy(logits_per_anchor, labels)
            loss_text = F.cross_entropy(logits_per_multi, labels)

        total_loss = (loss_text + loss_vision) / 2

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "loss_text": loss_text,
                "loss_vision": loss_vision,
                "logits_per_anchor": logits_per_anchor,
                "logits_per_multi": logits_per_multi,
            }

        return total_loss


class MultiEncoderClipLoss(nn.Module):
    """Multi-encoder CLIP Loss supporting M:N alignment.

    This loss supports scenarios with multiple encoders on both sides,
    e.g., M vision encoders and N text encoders.

    The loss computes all pairwise similarities and aggregates them.
    """

    def __init__(
        self,
        num_vision_encoders: int = 1,
        num_text_encoders: int = 1,
        aggregation: str = 'mean',
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        """Initialize MultiEncoderClipLoss.

        Args:
            num_vision_encoders: Number of vision encoders
            num_text_encoders: Number of text encoders
            aggregation: How to aggregate features ('mean', 'max')
            local_loss: Compute loss locally per GPU
            gather_with_grad: Gather features with gradient
            cache_labels: Cache ground truth labels
            rank: Current process rank
            world_size: Total number of processes
            use_horovod: Use Horovod for distributed training
        """
        super().__init__()
        self.num_vision_encoders = num_vision_encoders
        self.num_text_encoders = num_text_encoders
        self.aggregation = aggregation
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """Get ground truth labels."""
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(
        self,
        vision_features_list: List[torch.Tensor],
        text_features_list: List[torch.Tensor],
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        """Forward pass for multi-encoder loss.

        Args:
            vision_features_list: List of vision features from M encoders
            text_features_list: List of text features from N encoders
            logit_scale: Scale factor for logits
            logit_bias: Optional bias for logits
            output_dict: Whether to return dict with detailed outputs

        Returns:
            Loss value or dict
        """
        device = vision_features_list[0].device

        # Aggregate vision features
        if len(vision_features_list) > 1:
            vision_stacked = torch.stack(vision_features_list, dim=1)
            if self.aggregation == 'mean':
                vision_features = vision_stacked.mean(dim=1)
            elif self.aggregation == 'max':
                vision_features = vision_stacked.max(dim=1).values
        else:
            vision_features = vision_features_list[0]

        # Aggregate text features
        if len(text_features_list) > 1:
            text_stacked = torch.stack(text_features_list, dim=1)
            if self.aggregation == 'mean':
                text_features = text_stacked.mean(dim=1)
            elif self.aggregation == 'max':
                text_features = text_stacked.max(dim=1).values
        else:
            text_features = text_features_list[0]

        # Compute logits similar to standard CLIP
        if self.world_size > 1:
            from .loss import gather_features
            all_vision_features, all_text_features = gather_features(
                vision_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * vision_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_vision_features.T
            else:
                logits_per_image = logit_scale * all_vision_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * vision_features @ text_features.T
            logits_per_text = logit_scale * text_features @ vision_features.T

        if logit_bias is not None:
            logits_per_image = logits_per_image + logit_bias
            logits_per_text = logits_per_text + logit_bias

        # Get labels
        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        # Compute loss
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_image + loss_text) / 2

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "loss_image": loss_image,
                "loss_text": loss_text,
            }

        return total_loss
