from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _load_backbone_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint.get("model_state", checkpoint))
    return {key.removeprefix("module."): value for key, value in state.items()}


class RetCEVD(nn.Module):
    """Three-layer OCTA and tabular RetCEVD encoder/classifier."""

    def __init__(self, tabular_dim: int = 69, backbone_weights: Optional[Path] = None):
        super().__init__()
        backbone = models.resnet50(weights=None)
        if backbone_weights is not None:
            state = _load_backbone_state(Path(backbone_weights))
            result = backbone.load_state_dict(state, strict=False)
            unexpected = [key for key in result.unexpected_keys if not key.startswith("fc.")]
            missing = [key for key in result.missing_keys if not key.startswith("fc.")]
            if unexpected or missing:
                raise RuntimeError(
                    f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )

        self.image_stem = backbone.conv1
        self.image_encoder = nn.Sequential(
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.image_bottleneck = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        self.fusion = nn.MultiheadAttention(
            embed_dim=256,
            kdim=tabular_dim,
            vdim=tabular_dim,
            num_heads=4,
            batch_first=True,
        )
        self.classifier = nn.Linear(256, 2)
        self.tabular_dim = tabular_dim

    @property
    def representation_dim(self) -> int:
        return 128 + self.tabular_dim

    def forward(self, images: torch.Tensor, tabular: torch.Tensor):
        image_features = self.image_stem(images)
        image_features = self.image_encoder(image_features).flatten(1)
        projected_image = self.image_projection(image_features)
        representation = F.normalize(
            torch.cat([projected_image, tabular], dim=1), dim=-1
        )

        query = self.image_bottleneck(image_features).unsqueeze(1)
        tabular_token = tabular.unsqueeze(1)
        fusion_output, _ = self.fusion(
            query=query,
            key=tabular_token,
            value=tabular_token,
            need_weights=False,
        )
        fused = query.squeeze(1) + fusion_output.squeeze(1)
        return representation, self.classifier(fused)
