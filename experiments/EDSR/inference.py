import argparse
import os
import sys
import json
from time import time

from PIL import Image

import torch
import torchvision.transforms.functional as TF

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from models.edsr import EDSR, load_model
from models.utils import seed_everything
from models.ensemble import forward_x8

def parse_args():
    parser = argparse.ArgumentParser(description="EDSR Inference - Upscale image")
    
    # Loading model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    
    # Upscaling image
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default=None, help="Path to save upscaled image")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble (often leads to higher accuracy)")
    
    # Reproducibility
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("cuda not available, using cpu instead")
    
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    model = EDSR(
        in_channels=3,
        n_blocks=config["n_blocks"],
        n_features=config["n_features"],
        scale_factor=config["scale"],
        activation=config["act"],
        res_scale=config["res_scale"],
    ).to(args.device)

    load_model(model, args.checkpoint, load_tail=True, device=torch.device(args.device))
    model.eval()

    img = Image.open(args.img).convert("RGB")
    lr = TF.to_tensor(img).unsqueeze(0).to(args.device)
    start_time = time()
    with torch.no_grad():
        if args.use_ensemble:
            sr = forward_x8(model, lr)
        else:
            sr = model(lr)
    
        sr = sr.clamp(0.0, 1.0)
    end_time = time()
    elapsed = end_time - start_time

    sr_img = TF.to_pil_image(sr.squeeze(0).cpu())

    if args.out is None:
        base, ext = os.path.splitext(args.img)
        args.out = f"{base}_{config['scale']}x_upscaled{ext}"

    sr_img.save(args.out)
    print(f"Upscaled image saved to {args.out}")
    print(f"Time taken: {elapsed:.3f}s")

if __name__ == "__main__":
    main()
