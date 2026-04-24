"""
src/model.py: DRDModel architecture (EfficientNet-B3 + Channel/Spatial Attention).
Must match training exactly so the checkpoint loads without errors.
"""

import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        mid = max(in_ch // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch),
        )

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        desc = torch.cat(
            [x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1
        )
        return x * torch.sigmoid(self.conv(desc))


class DRDModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3", num_classes=5, dropout=0.3):
        super().__init__()
        self.cnn = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")
        with torch.no_grad():
            self.feat_dim = self.cnn(torch.zeros(1, 3, 64, 64)).shape[1]

        self.ch_attn  = ChannelAttention(self.feat_dim)
        self.sp_attn  = SpatialAttention()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim * 2, num_classes),
        )

    def forward(self, x):
        f = self.sp_attn(self.ch_attn(self.cnn(x)))
        pooled = torch.cat(
            [self.avg_pool(f).flatten(1), self.max_pool(f).flatten(1)], dim=1
        )
        return self.head(pooled)


def load_model(checkpoint_path: str, device: torch.device) -> DRDModel:
    """Load DRDModel from a .pt checkpoint onto *device*."""
    model = DRDModel(backbone="efficientnet_b3", num_classes=5, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Safely extract the weights depending on how the checkpoint was saved
    if "state" in checkpoint:
        weights = checkpoint["state"]
    elif "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        weights = checkpoint
        
    # Load the extracted weights into the model
    model.load_state_dict(weights)
    
    model.to(device)
    model.eval()
    return model