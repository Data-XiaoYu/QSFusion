import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from datasets import EvalPairTransform, PairedFusionFolderDataset
from modules import build_qsfusion
from options import test_options
from utils.checkpoint import load_qsfusion_checkpoint
from utils.image_utils import save_fused_tensor


def choose_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_qsfusion(model, checkpoint_path, device):
    _, message = load_qsfusion_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(message)


@torch.no_grad()
def main(args):
    device = choose_device(args)
    model = build_qsfusion(dim=args.base_channels).to(device)
    load_qsfusion(model, args.ckpt_path, device)
    model.eval()

    dataset = PairedFusionFolderDataset(args.vi_path, args.ir_path, transform=EvalPairTransform())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    output_dir = Path(args.out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    for step, (visible, infrared, name) in enumerate(loader, start=1):
        visible = visible.to(device)
        infrared = infrared.to(device)
        start = time.time()
        fused, *_ = model(infrared, visible)
        elapsed = time.time() - start
        total_time += elapsed
        output_path = save_fused_tensor(fused, output_dir, name[0])
        print(f"[{step}/{len(loader)}] saved {output_path} ({elapsed:.4f}s)")

    if len(loader) > 0:
        print(f"Average inference time: {total_time / len(loader):.4f}s/image")


if __name__ == "__main__":
    main(test_options().parse_args())
