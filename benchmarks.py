import json

import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from models.utils import calculate_psnr
from experiments.EDSR.dataset import load_dataset

def evaluate_interpolation(dataset, scale_factor=2, mode="bicubic"):
    total_mse = 0.0
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for lr_img, hr_img in dataset:
        if lr_img.dim() == 3:
            lr_img = lr_img.unsqueeze(0)
        if hr_img.dim() == 3:
            hr_img = hr_img.unsqueeze(0)
        h, w = lr_img.shape[-2:]
        new_h, new_w = h * scale_factor, w * scale_factor

        if mode in ["bilinear", "bicubic"]:
            upsampled = F.interpolate(lr_img, size=(new_h, new_w), mode=mode, align_corners=False)
        else:
            upsampled = F.interpolate(lr_img, size=(new_h, new_w), mode=mode)

        upsampled = upsampled.clamp(0, 1)

        mse = F.mse_loss(upsampled, hr_img, reduction="mean").item()
        l1 = F.l1_loss(upsampled, hr_img, reduction="mean").item()
        psnr_val = calculate_psnr(upsampled, hr_img, scale_factor=scale_factor).item()
        ssim_val = ssim_fn(upsampled, hr_img).item()

        total_mse += mse
        total_l1 += l1
        total_psnr += psnr_val
        total_ssim += ssim_val

    n_samples = len(dataset)
    return {
        "MSE": total_mse / n_samples,
        "L1": total_l1 / n_samples,
        "PSNR (dB)": total_psnr / n_samples,
        "SSIM": total_ssim / n_samples
    }

if __name__ == "__main__":
    all_results = {}
    for color_mode in ["RGB", "YCbCr"]:
        all_results[color_mode] = {}
        for dataset_name in ["Set5", "Set14", "BSD100", "Urban100", "DIV2K-val"]:    
            all_results[color_mode][dataset_name] = {}

            for scale_factor in [2, 3, 4]:
                key = f"{scale_factor}x"
                all_results[color_mode][dataset_name][key] = {}

                dataset = load_dataset(root_dir=f"./data/{dataset_name}", scale_factor=scale_factor, mode=color_mode)

                for interpolation in ["nearest", "bilinear", "bicubic"]:
                    results = evaluate_interpolation(dataset, scale_factor=scale_factor, mode=interpolation)
                    all_results[color_mode][dataset_name][key][interpolation] = results

    with open("benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=4)