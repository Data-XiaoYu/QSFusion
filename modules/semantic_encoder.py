import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedSemanticEncoder(nn.Module):
    """Shared semantic encoder used to compute S_ir, S_vi, and S_fused."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.semantic_stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.semantic_stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.semantic_stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.semantic_stage1(image)
        feature = self.semantic_stage2(feature)
        return self.semantic_stage3(feature)


def semantic_cosine_distance(feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
    batch = feature_a.size(0)
    feature_a = F.normalize(feature_a.view(batch, -1), dim=1)
    feature_b = F.normalize(feature_b.view(batch, -1), dim=1)
    return 1.0 - (feature_a * feature_b).sum(dim=1)
