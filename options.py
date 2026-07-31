import argparse


def add_common_options(parser: argparse.ArgumentParser):
    parser.add_argument("--device", default="cuda", help="Device name: cuda or cpu.")
    parser.add_argument("--gpu_id", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--base_channels", type=int, default=48, help="Base channel width.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    return parser


def train_options():
    parser = argparse.ArgumentParser(description="Train QSFusion.")
    add_common_options(parser)
    parser.add_argument("--dataset_root", required=True, help="Dataset root with train/ and optional test/.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--output_dir", default="./experiments/qsfusion")
    parser.add_argument("--init_weights", default="", help="Optional warm-start checkpoint.")
    parser.add_argument("--no_eval", action="store_true", help="Skip validation.")
    parser.add_argument("--lambda_semantic", type=float, default=0.0)
    parser.add_argument("--lambda_quality", type=float, default=0.0)
    return parser


def test_options():
    parser = argparse.ArgumentParser(description="Run QSFusion inference.")
    add_common_options(parser)
    parser.add_argument("--ckpt_path", default="./weights/qsfusion_final.pth")
    parser.add_argument("--vi_path", default="./test_data/vi")
    parser.add_argument("--ir_path", default="./test_data/ir")
    parser.add_argument("--out_path", default="./outputs")
    return parser
