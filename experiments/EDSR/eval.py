import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import torch.utils.data as data

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from models.edsr import EDSR, load_model
from models.utils import seed_everything
from .dataset import load_DIV2K_dataset
from .train import evaluate

def parse_args():
    # TODO: Add datasets: Set5, Set14 and Urban100
    parser = argparse.ArgumentParser(description="EDSR Evaluation")
    # Loading model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json used during training")
    
    # Evaluation
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    
    # Reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)
    
    for key, value in config_data.items():
        if hasattr(args, key) and key not in ["device", "checkpoint", "seed", "batch"]:
            setattr(args, key, value)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available, using cpu instead")
        args.device = "cpu"

    return args

def main():
    criterion = nn.L1Loss()
    args = parse_args()
    seed_everything(args.seed)

    print("Loading dataset...")
    train_dataset, val_dataset = load_DIV2K_dataset(
        args.data_dir,
        scale_factor=args.scale,
        patch_size=args.patch_size,
        transform=None
    )
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    print("Building model...")
    model = EDSR(
        in_channels=3,
        n_blocks=args.n_blocks,
        n_features=args.n_features,
        scale_factor=args.scale,
        activation=args.activation,
        residual_scaling=args.res_scale,
    ).to(args.device)

    print(f"Loading checkpoint from {args.checkpoint}")
    load_model(model, args.checkpoint, load_tail=True)

    print(f"Running evaluation on {args.device}...")
    train_loss, train_psnr, train_ssim_val = evaluate(model, criterion, train_loader, args.device)
    val_loss, val_psnr, val_ssim_val = evaluate(model, criterion, val_loader, args.device)
    
    print(f"\Results:")
    print(f"\tTrain: Loss={train_loss:.4f}, PSNR={train_psnr:.3f}dB, SSIM={train_ssim_val:.5f}")
    print(f"\tVal: Loss={val_loss:.4f}, PSNR={val_psnr:.3f}dB, SSIM={val_ssim_val:.5f}")
    
if __name__ == "__main__":
    main()
