import timm
import torch
import torch.nn as nn


class EfficientNetMLP(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3, unfreeze_blocks: int = 7):
        super().__init__()
        backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=1, num_classes=0
        )
        # Freeze everything, then selectively unfreeze
        for param in backbone.parameters():
            param.requires_grad = False

        # Always unfreeze the neck (conv_head, bn2)
        for part in (backbone.conv_head, backbone.bn2):
            for param in part.parameters():
                param.requires_grad = True

        # Unfreeze the last `unfreeze_blocks` block groups (0 = none, 7 = all)
        for i in range(7 - unfreeze_blocks, 7):
            for param in backbone.blocks[i].parameters():
                param.requires_grad = True

        # unfreeze_blocks=7 also unfreezes stem
        if unfreeze_blocks == 7:
            for part in (backbone.conv_stem, backbone.bn1):
                for param in part.parameters():
                    param.requires_grad = True

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
