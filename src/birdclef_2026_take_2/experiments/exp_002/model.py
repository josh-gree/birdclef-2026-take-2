import timm
import torch
import torch.nn as nn


class EfficientNetMLP(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3, backbone_variant: str = "efficientnet_b0"):
        super().__init__()
        backbone = timm.create_model(
            backbone_variant, pretrained=True, in_chans=1, num_classes=0
        )
        self.backbone = backbone
        feat_dim = backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
