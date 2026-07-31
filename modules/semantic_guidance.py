import torch
import torch.nn as nn
import torch.nn.functional as F


class ASGModule(nn.Module):
    """Adaptive semantic guidance module that predicts attention mask M."""

    def __init__(self, in_channels_ir: int = 3, in_channels_vi: int = 3, mid_channels: int = 32):
        super().__init__()
        self.ir_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels_ir, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.vis_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels_vi, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.mask_generation_network = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, infrared: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        infrared_edge = self.sobel_gradient_magnitude(infrared)
        visible_edge = self.sobel_gradient_magnitude(visible)

        enhanced_ir = infrared + infrared_edge
        enhanced_vis = visible + visible_edge

        ir_feature = self.ir_feature_extractor(enhanced_ir)
        vis_feature = self.vis_feature_extractor(enhanced_vis)
        attention_mask = self.mask_generation_network(torch.cat([ir_feature, vis_feature], dim=1))
        return torch.sigmoid(attention_mask)

    @staticmethod
    def sobel_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device
        ).view(1, 1, 3, 3)
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
