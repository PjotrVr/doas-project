import random
import os

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

class DIV2KCropDataset(data.Dataset):
    def __init__(self, lr_paths, hr_paths, patch_size=96, scale_factor=2):
        assert len(lr_paths) == len(hr_paths), "Mismatched number of images"
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.lr_patch_size = patch_size // scale_factor

    def __len__(self):
        return len(self.lr_imgs)

    def __getitem__(self, idx):
        hr = transforms.ToTensor()(Image.open(self.hr_paths[idx]))
        lr = transforms.ToTensor()(Image.open(self.lr_paths[idx]))
        _, h_lr, w_lr = lr.shape
        _, h_hr, w_hr = hr.shape

        if h_lr < self.lr_patch_size or w_lr < self.lr_patch_size:
            raise ValueError("LR image too small for crop")

        x_lr = random.randint(0, w_lr - self.lr_patch_size)
        y_lr = random.randint(0, h_lr - self.lr_patch_size)
        x_hr = x_lr * self.scale_factor
        y_hr = y_lr * self.scale_factor

        lr_patch = lr[:, y_lr:y_lr + self.lr_patch_size, x_lr:x_lr + self.lr_patch_size]
        hr_patch = hr[:, y_hr:y_hr + self.patch_size, x_hr:x_hr + self.patch_size]
        return lr_patch, hr_patch

def get_div2k_dataset(scale_factor=2, patch_size=96):
    train_hr_dir = "./data/DIV2K/DIV2K_train_HR"
    train_lr_dir = f"./data/DIV2K/DIV2K_train_LR_bicubic/X{scale_factor}"
    valid_hr_dir = "./data/DIV2K/DIV2K_valid_HR"
    valid_lr_dir = f"./data/DIV2K/DIV2K_valid_LR_bicubic/X{scale_factor}"

    hr_train_img_files = sorted([os.path.join(train_hr_dir, f) for f in os.listdir(train_hr_dir) if f.endswith(("png", "jpg"))])
    lr_train_img_files = sorted([os.path.join(train_lr_dir, f) for f in os.listdir(train_lr_dir) if f.endswith(("png", "jpg"))])
    hr_valid_img_files = sorted([os.path.join(valid_hr_dir, f) for f in os.listdir(valid_hr_dir) if f.endswith(("png", "jpg"))])
    lr_valid_img_files = sorted([os.path.join(valid_lr_dir, f) for f in os.listdir(valid_lr_dir) if f.endswith(("png", "jpg"))])

    train_dataset = DIV2KCropDataset(lr_train_img_files, hr_train_img_files, patch_size=patch_size, scale_factor=scale_factor)
    valid_dataset = DIV2KCropDataset(lr_valid_img_files, hr_valid_img_files, patch_size=patch_size, scale_factor=scale_factor)
    return train_dataset, valid_dataset