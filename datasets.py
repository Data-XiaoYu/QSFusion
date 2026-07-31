import os
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_pair_dirs(root: str | os.PathLike):
    root = Path(root)
    candidates = [
        (root / "vi", root / "ir"),
        (root / "Visible", root / "Infrared"),
        (root / "visible", root / "infrared"),
    ]
    for visible_dir, infrared_dir in candidates:
        if visible_dir.exists() and infrared_dir.exists():
            return visible_dir, infrared_dir
    raise FileNotFoundError(
        f"Expected paired image folders under {root}: vi/ir or Visible/Infrared."
    )


def collect_image_pairs(root: str | os.PathLike):
    visible_dir, infrared_dir = _find_pair_dirs(root)
    visible_paths = sorted(
        p for p in visible_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    infrared_paths = sorted(
        p for p in infrared_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(visible_paths) != len(infrared_paths):
        raise ValueError(
            f"Visible and infrared image counts differ: {len(visible_paths)} vs {len(infrared_paths)}."
        )
    return visible_paths, infrared_paths


class TrainPairTransform:
    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, visible, infrared):
        width, height = visible.size
        crop_h = crop_w = self.crop_size
        if width < crop_w or height < crop_h:
            pad_w = max(crop_w - width, 0)
            pad_h = max(crop_h - height, 0)
            visible = TF.pad(visible, (0, 0, pad_w, pad_h), fill=0)
            infrared = TF.pad(infrared, (0, 0, pad_w, pad_h), fill=0)

        top, left, height, width = transforms.RandomCrop.get_params(
            visible, output_size=(crop_h, crop_w)
        )
        visible = TF.crop(visible, top, left, height, width)
        infrared = TF.crop(infrared, top, left, height, width)

        if torch.rand(1).item() < 0.5:
            visible = TF.hflip(visible)
            infrared = TF.hflip(infrared)
        if torch.rand(1).item() < 0.5:
            visible = TF.vflip(visible)
            infrared = TF.vflip(infrared)

        return TF.to_tensor(visible), TF.to_tensor(infrared)


class EvalPairTransform:
    def __call__(self, visible, infrared):
        width, height = visible.size
        new_width = max((width // 16) * 16, 16)
        new_height = max((height // 16) * 16, 16)
        visible = TF.resize(visible, (new_height, new_width))
        infrared = TF.resize(infrared, (new_height, new_width))
        return TF.to_tensor(visible), TF.to_tensor(infrared)


class PairedFusionDataset(Dataset):
    def __init__(self, root: str | os.PathLike, transform=None):
        self.visible_paths, self.infrared_paths = collect_image_pairs(root)
        self.transform = transform

    def __len__(self):
        return len(self.visible_paths)

    def __getitem__(self, index):
        visible_path = self.visible_paths[index]
        infrared_path = self.infrared_paths[index]
        visible = Image.open(visible_path).convert("RGB")
        infrared = Image.open(infrared_path).convert("RGB")

        if self.transform is not None:
            visible, infrared = self.transform(visible, infrared)
        else:
            visible, infrared = TF.to_tensor(visible), TF.to_tensor(infrared)

        return visible, infrared, visible_path.stem


class PairedFusionFolderDataset(Dataset):
    def __init__(self, visible_dir: str | os.PathLike, infrared_dir: str | os.PathLike, transform=None):
        visible_dir = Path(visible_dir)
        infrared_dir = Path(infrared_dir)
        self.visible_paths = sorted(
            p for p in visible_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.infrared_paths = sorted(
            p for p in infrared_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(self.visible_paths) != len(self.infrared_paths):
            raise ValueError(
                f"Visible and infrared image counts differ: {len(self.visible_paths)} vs {len(self.infrared_paths)}."
            )
        self.transform = transform

    def __len__(self):
        return len(self.visible_paths)

    def __getitem__(self, index):
        visible_path = self.visible_paths[index]
        infrared_path = self.infrared_paths[index]
        visible = Image.open(visible_path).convert("RGB")
        infrared = Image.open(infrared_path).convert("RGB")
        if self.transform is not None:
            visible, infrared = self.transform(visible, infrared)
        else:
            visible, infrared = TF.to_tensor(visible), TF.to_tensor(infrared)
        return visible, infrared, visible_path.stem
