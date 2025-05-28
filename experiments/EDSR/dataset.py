import os
import random
import math

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

class SRDataset(data.Dataset):
    def __init__(self, lr_paths, hr_paths, scale_factor, patch_size=None, transform=None, mode="rgb"):
        super().__init__()
        assert len(lr_paths) == len(hr_paths), "Number of LR and HR images must be the same"
        self.mode = mode.lower()
        assert self.mode in ["rgb", "ycbcr"], "Mode must be either RGB or YCbCr"

        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])
        if self.mode == "ycbcr":
            lr_img = lr_img.convert("YCbCr")
            hr_img = hr_img.convert("YCbCr")

        if self.patch_size:
            hr_patch_size = self.patch_size * self.scale_factor
            lr_w, lr_h = lr_img.size
            lr_x = random.randint(0, lr_w - self.patch_size)
            lr_y = random.randint(0, lr_h - self.patch_size)
            hr_x = lr_x * self.scale_factor
            hr_y = lr_y * self.scale_factor

            lr_img = lr_img.crop((lr_x, lr_y, lr_x + self.patch_size, lr_y + self.patch_size))
            hr_img = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)
        else:
            lr_img = TF.to_tensor(lr_img)
            hr_img = TF.to_tensor(hr_img)
        return lr_img, hr_img

def load_DIV2K_dataset(data_dir="./data", scale_factor=2, patch_size=48, transform=None, mode="rgb"):
    train_hr_dir = os.path.join(data_dir, "DIV2K_train_HR")
    train_lr_dir = os.path.join(data_dir, f"DIV2K_train_LR_bicubic/X{scale_factor}")
    val_hr_dir = os.path.join(data_dir, "DIV2K_valid_HR")
    val_lr_dir = os.path.join(data_dir, f"DIV2K_valid_LR_bicubic/X{scale_factor}")

    hr_train_img_files = sorted([os.path.join(train_hr_dir, f) for f in os.listdir(train_hr_dir) if f.endswith(("png", "jpg"))])
    lr_train_img_files = sorted([os.path.join(train_lr_dir, f) for f in os.listdir(train_lr_dir) if f.endswith(("png", "jpg"))])
    hr_val_img_files = sorted([os.path.join(val_hr_dir, f) for f in os.listdir(val_hr_dir) if f.endswith(("png", "jpg"))])
    lr_val_img_files = sorted([os.path.join(val_lr_dir, f) for f in os.listdir(val_lr_dir) if f.endswith(("png", "jpg"))])

    # hr_train_img_files = hr_train_img_files
    # lr_train_img_files = lr_train_img_files
    # hr_val_img_files = hr_val_img_files[:20]
    # lr_val_img_files = lr_val_img_files[:20]
    
    train_dataset = SRDataset(lr_train_img_files, hr_train_img_files, patch_size=patch_size, scale_factor=scale_factor, mode=mode, transform=transform)
    val_dataset = SRDataset(lr_val_img_files, hr_val_img_files, patch_size=patch_size, scale_factor=scale_factor, mode=mode, transform=None)
    return train_dataset, val_dataset

class PairedRandomTransform:
    def __init__(self, hflip=True, rot=True):
        self.hflip = hflip
        self.rot = rot

    def __call__(self, lr, hr):
        if self.hflip and random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        if self.rot:
            k = random.randint(0, 3)
            if k:
                lr = TF.rotate(lr, angle=90*k)
                hr = TF.rotate(hr, angle=90*k)
        lr = TF.to_tensor(lr)
        hr = TF.to_tensor(hr)
        return lr, hr