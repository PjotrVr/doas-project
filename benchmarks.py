import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="mean")
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-10))

def evaluate_interpolation(lr_paths, hr_paths, scale_factor=2, color_mode="RGB", mode="bicubic"):
    assert len(lr_paths) == len(hr_paths)
    to_tensor = transforms.ToTensor()
    
    total_mse = 0.0
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for lr_path, hr_path in zip(lr_paths, hr_paths):
        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        if color_mode == "Y":
            lr_img = lr_img.convert("YCbCr").split()[0]
            hr_img = hr_img.convert("YCbCr").split()[0]

        lr_tensor = to_tensor(lr_img).unsqueeze(0)  # [1, C, H, W] or [1, 1, H, W]
        hr_tensor = to_tensor(hr_img).unsqueeze(0)

        _, _, h_lr, w_lr = lr_tensor.shape
        new_h, new_w = h_lr * scale_factor, w_lr * scale_factor

        if mode in ["bilinear", "bicubic"]:
            upsampled = F.interpolate(lr_tensor, size=(new_h, new_w), mode=mode, align_corners=False)
        else:
            upsampled = F.interpolate(lr_tensor, size=(new_h, new_w), mode=mode)
            
        # upsampled = F.interpolate(lr_tensor, size=(new_h, new_w), mode=mode, align_corners=False).clamp(0, 1)
        upsampled = upsampled.clamp(0, 1)
        
        mse = F.mse_loss(upsampled, hr_tensor, reduction="mean").item()
        l1 = F.l1_loss(upsampled, hr_tensor, reduction="mean").item()
        psnr_val = psnr(upsampled, hr_tensor).item()
        ssim_val = ssim_fn(upsampled, hr_tensor).item()

        total_mse += mse
        total_l1 += l1
        total_psnr += psnr_val
        total_ssim += ssim_val

    N = len(lr_paths)
    return {
        "MSE": total_mse / N,
        "L1": total_l1 / N,
        "PSNR (dB)": total_psnr / N,
        "SSIM": total_ssim / N
    }

for split in ["train", "valid"]:
    hr_dir = f"./data/DIV2K_{split}_HR"
    hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(("png", "jpg"))])
    for scale_factor in [2, 3, 4]:
        lr_dir = f"./data/DIV2K_{split}_LR_bicubic/X{scale_factor}"
        lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(("png", "jpg"))])    
        for mode in ["nearest", "bilinear", "bicubic"]:
            for color_mode in ["RGB", "Y"]:
                results = evaluate_interpolation(lr_paths, hr_paths, scale_factor=scale_factor, color_mode=color_mode, mode=mode)
                print(f"{split=}, {mode=}, {scale_factor=}, {color_mode=}")
                for k, v in results.items():
                    print(f"{k}: {v:.4f}")
                    