import random
import os

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_mean(img_dir, transform, in_channels=3):
    channel_sum = torch.zeros(in_channels, dtype=torch.float64)
    channel_squared_sum = torch.zeros(in_channels, dtype=torch.float64)
    n_pixels = 0
    for img_file in os.listdir(img_dir):
        img = transform(Image.open(os.path.join(img_dir, img_file)))
        channel_sum += img.sum(dim=(1, 2))
        channel_squared_sum += (img ** 2).sum(dim=(1, 2))
        n_pixels += img.shape[1] * img.shape[2]

    channel_mean = channel_sum / n_pixels
    channel_std = (channel_squared_sum / n_pixels - channel_mean**2).sqrt()
    return channel_mean, channel_std

def calculate_psnr(sr, hr, scale_factor):
    shave = scale_factor + 6
    sr_crop = sr[..., shave:-shave, shave:-shave]
    hr_crop = hr[..., shave:-shave, shave:-shave]
    mse = F.mse_loss(sr_crop, hr_crop)
    return -10 * torch.log10(mse)

def is_img_file(path):
    return os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg"))
