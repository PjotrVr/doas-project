import os
import random
import sys
import math
import argparse
from datetime import datetime
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from .dataset import load_dataset
from models.edsr import EDSR, save_model, load_model
from models.utils import calculate_psnr, seed_everything
from models.ensemble import forward_x8

def parse_args():
    parser = argparse.ArgumentParser(description="EDSR Training")

    # Directories
    parser.add_argument("--train_dir", type=str, default="./data-augmented/DIV2K-train", help="Path to train data directory")
    parser.add_argument("--val_dir", type=str, default="./data/DIV2K-val", help="Path to validation data directory")
    parser.add_argument("--run_dir", type=str, default="runs", help="Directory to save logs/models")
    
    # Model parameters
    parser.add_argument("--scale", type=int, choices=[2, 3, 4], default=2, help="Super-resolution scale factor")
    parser.add_argument("--n_blocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("--n_features", type=int, default=64, help="Number of feature maps in the model")
    parser.add_argument("--res_scale", type=float, default=1.0, help="Residual scaling factor")
    parser.add_argument("--act", type=str, choices=["relu", "prelu"], default="relu", help="Activation function")
    
    # Training parameters
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--val_freq", type=int, default=1, help="Validate every N epochs")
    
    # Reproducibility and system
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Resume training/pretraining    
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume full training from")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to checkpoint to load weights but reset tail")
    parser.add_argument("--load_tail", action="store_true", help="Whether to load tail (used with --pretrain)")

    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, using cpu instead")

    return args

def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
    # total_samples = 0
    for lr_batch, hr_batch in loader:
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        sr_batch = model(lr_batch)
        sr_batch = torch.clamp(sr_batch, 0.0, 1.0)
        optimizer.zero_grad()

        loss = criterion(sr_batch, hr_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += calculate_psnr(sr_batch, hr_batch, model.scale_factor).item()
        total_ssim += ssim(sr_batch, hr_batch, data_range=1.0).item()
        # total_samples += lr_batch.shape[0]

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    return avg_loss, avg_psnr, avg_ssim

def evaluate(model, criterion, loader, device, ensemble=False):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
    n_samples = 0
    with torch.no_grad():
        for lr_img, hr_img in loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            if ensemble:
                sr_img = forward_x8(model, lr_img)
            else:
                sr_img = model(lr_img)
            
            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            loss = criterion(sr_img, hr_img)

            total_loss += loss.item()
            total_psnr += calculate_psnr(sr_img, hr_img, model.scale_factor).item()
            total_ssim += ssim(sr_img, hr_img, data_range=1.0).item()
            n_samples += 1

    avg_loss = total_loss / n_samples #len(dataset)
    avg_psnr = total_psnr / n_samples #len(dataset)
    avg_ssim = total_ssim / n_samples #len(dataset)
    model.train()
    return avg_loss, avg_psnr, avg_ssim

def main():
    args = parse_args()

    train_dataset = load_dataset(args.train_dir, scale_factor=args.scale)
    val_dataset = load_dataset(args.val_dir, scale_factor=args.scale)
    print(len(train_dataset), len(val_dataset))
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    seed_everything(args.seed)
    model = EDSR(
        in_channels=3,
        n_blocks=args.n_blocks,
        n_features=args.n_features,
        scale_factor=args.scale,
        activation=args.act,
        res_scale=args.res_scale,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    run_name = f"EDSR-DIV2K-{timestamp}"
    run_dir = os.path.join(args.run_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    n_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of parameters:")
    print(f"\tLearnable: {n_learnable_params:,}" )
    print(f"\tFrozen: {n_frozen_params:,}" )
    print(f"\tTotal: {n_learnable_params + n_frozen_params:,}" )

    if args.resume:
        print(f"Resuming training from {args.resume}...")
        load_model(model, args.checkpoint, load_tail=True, device=torch.device(args.device))
    elif args.pretrain:
        print(f"Loading pretrained model from {args.pretrain} (load_tail={args.load_tail})...")
        load_model(model, args.checkpoint, load_tail=True, device=torch.device(args.device))
    
    print(f"Starting training on {args.device}...")

    seed_everything(args.seed)
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "train_psnr": [],
        "train_ssim": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_epochs": []
    }

    for epoch in range(args.epochs):
        train_loss, train_psnr, train_ssim = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["train_ssim"].append(train_ssim)
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"\tTrain: Loss={train_loss:.4f}, PSNR={train_psnr:.3f}dB, SSIM={train_ssim:.5f}")

        if epoch % args.val_freq == 0 or (epoch + 1) == args.epochs:
            val_loss, val_psnr, val_ssim = evaluate(model, val_loader, val_dataset, args.device)
            scheduler.step(val_loss)
            history["val_loss"].append(val_loss)
            history["val_psnr"].append(val_psnr)
            history["val_ssim"].append(val_ssim)
            history["val_epochs"].append(epoch)

            print(f"\tVal: Loss={val_loss:.4f}, PSNR={val_psnr:.3f}dB, SSIM={val_ssim:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("\tFound best model! Saving...")
                save_model(model, os.path.join(checkpoints_dir, "best.pt"))
            
        save_model(model, os.path.join(checkpoints_dir, f"epoch={epoch}.pt"))
    
    # Save all metrics in case we need it for later analysis
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"Finished training")

if __name__ == "__main__":
    main()