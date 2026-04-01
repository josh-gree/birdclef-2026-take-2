import timm
import torch
import torch.nn as nn


class GeMPooling(nn.Module):
    """Generalised Mean Pooling over the frequency axis.

    Collapses dim=2 (H=freq) of a (B, C, H', W') feature map, leaving
    the time axis (W') intact for the attention head downstream.
    p=1 → average pool, p→∞ → max pool. Initialised to p=3.
    """

    def __init__(self, init_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, H', W')
        p = self.p.clamp(min=1.0)
        return h.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)
        # returns (B, C, W') = (B, C, T)


class EfficientNetSpatialAttention(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3, backbone_variant: str = "efficientnet_b0"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_variant, pretrained=True, in_chans=1, num_classes=0
        )
        feat_dim = self.backbone.num_features

        self.freq_pool = GeMPooling()

        # Shared per-timestep head; nn.Linear broadcasts over leading dims
        # [B, T, feat_dim] → [B, T, num_classes]
        self.timestep_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Attention network: [B, T, feat_dim] → [B, T, 1]
        self.attn_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 256, 256) spectrogram
        feats = self.backbone.forward_features(x)    # (B, 1280, 8, 8)
        pooled = self.freq_pool(feats)                # (B, 1280, 8)
        time_feats = pooled.permute(0, 2, 1)          # (B, 8, 1280)

        step_logits = self.timestep_head(time_feats)  # (B, 8, num_classes)
        attn_scores = self.attn_net(time_feats)        # (B, 8, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return (attn_weights * step_logits).sum(dim=1)  # (B, num_classes)
