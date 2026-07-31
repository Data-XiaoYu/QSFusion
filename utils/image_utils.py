from pathlib import Path

from torchvision.transforms import functional as TF


def save_fused_tensor(image_tensor, output_dir, image_name: str, suffix: str = ".png"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = image_tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    output_path = output_dir / f"{image_name}{suffix}"
    TF.to_pil_image(image).save(output_path)
    return output_path
