import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from datasets import EvalPairTransform, PairedFusionDataset, TrainPairTransform
from modules import build_qsfusion
from options import train_options
from utils.checkpoint import load_qsfusion_checkpoint
from utils.train_utils import QSFusionLoss


def choose_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    _, message = load_qsfusion_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(message)


def run_epoch(model, criterion, loader, device, optimizer=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    last_log_time = time.time()

    for step, (visible, infrared, _) in enumerate(loader, start=1):
        visible = visible.to(device, non_blocking=True)
        infrared = infrared.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            fused, s_ir, s_vi, s_fused, q_pred, q_target = model(infrared, visible)
            losses = criterion(
                visible=visible,
                infrared=infrared,
                fused=fused,
                semantic_visible=s_vi,
                semantic_infrared=s_ir,
                semantic_fused=s_fused,
                quality_pred=q_pred,
                quality_target=q_target,
            )
            loss = losses["total"]

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        now = time.time()
        if now - last_log_time > 10:
            phase = "train" if training else "eval"
            print(f"[{phase} epoch {epoch}] step {step}/{len(loader)} loss={total_loss / step:.4f}")
            last_log_time = now

    return total_loss / max(len(loader), 1)


def main(args):
    device = choose_device(args)
    train_root = Path(args.dataset_root) / "train"
    test_root = Path(args.dataset_root) / "test"

    train_set = PairedFusionDataset(train_root, transform=TrainPairTransform(args.crop_size))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    eval_loader = None
    if not args.no_eval:
        if test_root.exists():
            eval_set = PairedFusionDataset(test_root, transform=EvalPairTransform())
            eval_loader = DataLoader(
                eval_set,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
        else:
            print(f"[Info] No test split found at {test_root}; validation is skipped.")

    model = build_qsfusion(dim=args.base_channels).to(device)
    load_checkpoint(model, args.init_weights, device)
    criterion = QSFusionLoss(
        lambda_semantic=args.lambda_semantic,
        lambda_quality=args.lambda_quality,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = run_epoch(model, criterion, train_loader, device, optimizer=optimizer, epoch=epoch)
        if eval_loader is None:
            val_loss = train_loss
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        else:
            with torch.no_grad():
                val_loss = run_epoch(model, criterion, eval_loader, device, epoch=epoch)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = output_dir / "qsfusion_best.pth"
            torch.save({"model": model.state_dict(), "args": vars(args)}, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main(train_options().parse_args())
