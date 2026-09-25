"""End-to-End Partial Backbone Unfreezing Model for Audio Deepfake Detection.

Combines HuggingFace Wav2Vec2Model (XLS-R 300M) and the Selective Layer
Summarization (SLS) Head into a single PyTorch nn.Module. Supports freezing
lower layers while leaving the top N transformer layers and SLS Head trainable.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .model import (
    INPUT_LENGTH,
    SLSHead,
    load_sls_checkpoint,
    pad_or_truncate,
)

logger = logging.getLogger(__name__)


class PartialUnfreezeSLSModel(nn.Module):
    """End-to-end model combining XLS-R 300M backbone and SLS head.

    Allows freezing lower transformer layers (e.g., layers 1-20) while
    unfreezing top transformer layers (e.g., layers 21-24) and the SLS head.
    Takes raw audio waveforms as input and outputs class log-probabilities.
    """

    def __init__(
        self,
        unfreeze_layers: int = 4,
        pretrained_head_version: str = "v1",
        gradient_checkpointing: bool = False,
    ):
        """Initialize the model.

        Args:
            unfreeze_layers: Number of top transformer layers to unfreeze (0 to 24).
                             0 = fully frozen backbone (identical to baseline).
                             4 = layers 21-24 unfrozen, 1-20 frozen.
            pretrained_head_version: 'v1' or 'v2' SLS checkpoint to initialize head.
            gradient_checkpointing: Enable HF gradient checkpointing on backbone to save VRAM.
        """
        super().__init__()
        self.unfreeze_layers = unfreeze_layers

        # 1. Load XLS-R 300M Backbone from HuggingFace
        logger.info("Loading XLS-R 300M backbone from HuggingFace...")
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on XLS-R backbone.")

        # 2. Load SLS Head and initialize with pretrained weights
        self.sls_head = SLSHead()
        load_sls_checkpoint(self.sls_head, checkpoint_version=pretrained_head_version)

        # 3. Configure layer-wise freezing
        self._configure_freezing(unfreeze_layers)

    def _configure_freezing(self, unfreeze_layers: int) -> None:
        """Freeze feature extractor and lower layers, unfreeze top N layers."""
        # A. Freeze CNN feature extractor completely
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, "feature_projection"):
            for param in self.backbone.feature_projection.parameters():
                param.requires_grad = False

        total_layers = len(self.backbone.encoder.layers)  # 24 layers in XLS-R 300M
        freeze_up_to = total_layers - unfreeze_layers

        logger.info(
            f"Configuring backbone freezing: {total_layers} total layers. "
            f"Freezing layers 1 to {freeze_up_to}, unfreezing top {unfreeze_layers} layers "
            f"(layers {freeze_up_to + 1} to {total_layers})."
        )

        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < freeze_up_to:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        # SLS Head parameters are always trainable
        for param in self.sls_head.parameters():
            param.requires_grad = True

        # Log trainable vs frozen counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        logger.info(
            f"Parameters: {trainable / 1e6:.2f}M trainable, {frozen / 1e6:.2f}M frozen "
            f"({trainable / (trainable + frozen) * 100:.1f}% trainable)."
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and SLS head.

        Args:
            waveforms: (B, T) raw audio waveform tensor, shape (B, 64600) @ 16kHz.

        Returns:
            log_probs: (B, 2) log-probabilities [spoof, bonafide].
        """
        # Pass through XLS-R backbone requesting all intermediate layer states
        outputs = self.backbone(waveforms, output_hidden_states=True)

        # outputs.hidden_states is tuple of 25: index 0 is CNN output, 1..24 are transformer layers
        hidden_states_list = list(outputs.hidden_states[1:])  # 24 tensors, each (B, T_frames=201, 1024)

        # Pass layer representations to SLS Head
        log_probs = self.sls_head(hidden_states_list)
        return log_probs

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-6,
        head_lr: float = 1e-5,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        """Create differential parameter groups for optimizer.

        Applies smaller learning rate to unfrozen foundation backbone layers and
        higher learning rate to the SLS classification head.
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone_unfrozen",
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "sls_head",
            })

        logger.info(
            f"Optimizer parameter groups created: "
            f"{len(backbone_params)} backbone tensors (lr={backbone_lr}), "
            f"{len(head_params)} head tensors (lr={head_lr})."
        )
        return groups
