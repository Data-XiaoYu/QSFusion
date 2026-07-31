import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.semantic_encoder import semantic_cosine_distance


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    if image.size(1) == 1:
        return image
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    return gray


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.stack((y, cb, cr), dim=1)


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    image = F.pad(image, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(image, kernel_x)
    grad_y = F.conv2d(image, kernel_y)
    return torch.abs(grad_x), torch.abs(grad_y)


def ssim_loss(image_a: torch.Tensor, image_b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = image_a.size(1)
    window_size = min(window_size, image_a.size(2), image_a.size(3))
    coords = torch.arange(window_size, dtype=image_a.dtype, device=image_a.device)
    coords = coords - window_size // 2
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] @ gaussian[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    mu_a = F.conv2d(image_a, window, groups=channels)
    mu_b = F.conv2d(image_b, window, groups=channels)
    sigma_a = F.conv2d(image_a * image_a, window, groups=channels) - mu_a.pow(2)
    sigma_b = F.conv2d(image_b * image_b, window, groups=channels) - mu_b.pow(2)
    sigma_ab = F.conv2d(image_a * image_b, window, groups=channels) - mu_a * mu_b

    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a.pow(2) + mu_b.pow(2) + c1) * (sigma_a + sigma_b + c2)
    )
    return 1.0 - ssim.mean()


class QSFusionLoss(nn.Module):
    def __init__(
        self,
        lambda_semantic: float = 0.0,
        lambda_quality: float = 0.0,
        intensity_weight: float = 4.0,
        ssim_weight: float = 1.0,
        color_weight: float = 12.0,
        gradient_weight: float = 10.0,
    ):
        super().__init__()
        self.lambda_semantic = lambda_semantic
        self.lambda_quality = lambda_quality
        self.intensity_weight = intensity_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.gradient_weight = gradient_weight

    def forward(
        self,
        visible,
        infrared,
        fused,
        semantic_visible,
        semantic_infrared,
        semantic_fused,
        quality_pred=None,
        quality_target=None,
    ):
        visible_gray = rgb_to_gray(visible)
        infrared_gray = rgb_to_gray(infrared)
        fused_gray = rgb_to_gray(fused)

        selection = (infrared_gray > visible_gray).float()
        intensity_target = selection * infrared + (1.0 - selection) * visible
        loss_intensity = F.l1_loss(fused, intensity_target)

        loss_ssim = ssim_loss(visible, fused) + ssim_loss(infrared_gray, fused_gray)

        visible_ycbcr = rgb_to_ycbcr(visible)
        fused_ycbcr = rgb_to_ycbcr(fused)
        loss_color = F.l1_loss(visible_ycbcr[:, 1:], fused_ycbcr[:, 1:])

        grad_vis_x, grad_vis_y = gradient_map(visible_gray)
        grad_ir_x, grad_ir_y = gradient_map(infrared_gray)
        grad_fused_x, grad_fused_y = gradient_map(fused_gray)
        loss_gradient = F.l1_loss(grad_fused_x, torch.maximum(grad_vis_x, grad_ir_x))
        loss_gradient = loss_gradient + F.l1_loss(grad_fused_y, torch.maximum(grad_vis_y, grad_ir_y))

        loss_semantic = (
            0.5 * semantic_cosine_distance(semantic_fused, semantic_infrared)
            + 0.5 * semantic_cosine_distance(semantic_fused, semantic_visible)
            - 0.1 * semantic_cosine_distance(semantic_infrared, semantic_visible)
        ).mean()

        if quality_pred is None or quality_target is None:
            loss_quality = fused.new_tensor(0.0)
        else:
            loss_quality = torch.mean((quality_pred - quality_target) ** 2)

        fusion_loss = (
            self.intensity_weight * loss_intensity
            + self.ssim_weight * loss_ssim
            + self.color_weight * loss_color
            + self.gradient_weight * loss_gradient
        )
        total = fusion_loss + self.lambda_semantic * loss_semantic + self.lambda_quality * loss_quality
        return {
            "total": total,
            "fusion": fusion_loss,
            "semantic": loss_semantic,
            "quality": loss_quality,
            "ssim": loss_ssim,
            "intensity": loss_intensity,
            "color": loss_color,
            "gradient": loss_gradient,
        }
