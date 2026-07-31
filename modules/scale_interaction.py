import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDimBlock(nn.Module):
    """Spatial enhancement plus channel attention used inside the MCI module."""

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        spatial_feature = self.spatial_enhancement(feature)
        channel_weight = self.channel_attention(spatial_feature)
        return feature + spatial_feature * channel_weight


class MCIModule(nn.Module):
    """Multi-scale cross-dimensional interaction module in Figure 5."""

    def __init__(self, channels: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.scale0_cross_dim = CrossDimBlock(channels)
        self.scale1_cross_dim = CrossDimBlock(channels)
        self.scale2_cross_dim = CrossDimBlock(channels)
        self.scale_fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape

        scale0_feature = self.scale0_cross_dim(feature)

        scale1_feature = F.interpolate(
            feature, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        scale1_feature = self.scale1_cross_dim(scale1_feature)
        scale1_feature = F.interpolate(
            scale1_feature, size=(height, width), mode="bilinear", align_corners=False
        )

        scale2_feature = F.interpolate(
            feature, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        scale2_feature = self.scale2_cross_dim(scale2_feature)
        scale2_feature = F.interpolate(
            scale2_feature, size=(height, width), mode="bilinear", align_corners=False
        )

        fused_feature = torch.cat([scale0_feature, scale1_feature, scale2_feature], dim=1)
        fused_feature = self.scale_fusion(fused_feature)
        return feature + self.gamma * fused_feature
