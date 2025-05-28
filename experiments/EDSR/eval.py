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
from .dataset import load_dataset
from .train import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="EDSR Evaluation")
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default="DIV2K-val", help="Path to data directory")
    
    # Loading model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json used during training")
    
    # Evaluation
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble (often leads to higher accuracy)")
    
    # Reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)
    
    for key, value in config_data.items():
        if hasattr(args, key) and key not in ["device", "checkpoint", "seed", "batch", "data_dir", "run_dir"]:
            setattr(args, key, value)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda not available, using cpu instead")
        args.device = "cpu"

    return args

def main():
    criterion = nn.L1Loss()
    args = parse_args()
    seed_everything(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    print("Loading dataset...")
    dataset = load_dataset(args.data_dir, scale_factor=config["scale"])
    
    print("Building model...")
    model = EDSR(
        in_channels=3,
        n_blocks=config["n_blocks"],
        n_features=config["n_features"],
        scale_factor=config["scale"],
        activation=config["activation"],
        res_scale=config["res_scale"],
    ).to(args.device)

    print(f"Loading checkpoint from {args.checkpoint}")
    load_model(model, args.checkpoint, load_tail=True, device=torch.device(args.device))

    print(f"Running evaluation on {args.device}...")
    loss, psnr, ssim_val = evaluate(model, criterion, dataset, args.device, ensemble=args.use_ensemble)

    print(f"Loss={loss:.4f}, PSNR={psnr:.3f}dB, SSIM={ssim_val:.5f}")
    
if __name__ == "__main__":
    main()
