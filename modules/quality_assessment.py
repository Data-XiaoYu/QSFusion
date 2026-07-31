import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_energy(image: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6).mean(dim=[1, 2, 3])


def local_contrast(image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    pad = kernel_size // 2
    mean = F.avg_pool2d(F.pad(gray, (pad, pad, pad, pad), mode="reflect"), kernel_size)
    variance = F.avg_pool2d(
        F.pad(gray**2, (pad, pad, pad, pad), mode="reflect"), kernel_size
    ) - mean**2
    return torch.sqrt(torch.clamp(variance, min=1e-6)).mean(dim=[1, 2, 3])


def soft_entropy(image: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    batch = gray.size(0)
    gray = gray.view(batch, -1)
    gray = (gray - gray.min(dim=1, keepdim=True)[0]) / (
        gray.max(dim=1, keepdim=True)[0] - gray.min(dim=1, keepdim=True)[0] + 1e-6
    )
    centers = torch.linspace(0.0, 1.0, steps=num_bins, dtype=gray.dtype, device=gray.device)
    centers = centers.view(1, num_bins, 1)
    logits = -((gray.unsqueeze(1) - centers) ** 2) * 50.0
    histogram = torch.softmax(logits, dim=1).mean(dim=2)
    return -(histogram * torch.log(histogram + 1e-6)).sum(dim=1)


class ReferenceFreeQualityEstimator(nn.Module):
    """Q_net in the reference-free quality self-evolution branch."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.quality_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.quality_regressor = nn.Linear(base_channels * 4, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        quality_feature = self.quality_feature_extractor(image)
        quality_feature = quality_feature.view(quality_feature.size(0), -1)
        return self.quality_regressor(quality_feature).squeeze(1)


def reference_free_quality_target(
    image: torch.Tensor,
    w_grad: float = 1.2,
    w_contrast: float = 1.2,
    w_entropy: float = 0.6,
) -> torch.Tensor:
    return (
        w_grad * gradient_energy(image)
        + w_contrast * local_contrast(image)
        + w_entropy * soft_entropy(image)
    )
