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
from models.utils import is_img_file

class SRDataset(data.Dataset):
    def __init__(self, lr_paths, hr_paths, mode="rgb"):
        super().__init__()
        assert len(lr_paths) == len(hr_paths), "Number of LR and HR images must be the same"
        self.mode = mode.lower()
        assert self.mode in ["rgb", "ycbcr"], "Mode must be either RGB or YCbCr"

        self.lr_paths = lr_paths
        self.hr_paths = hr_paths

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])
        if self.mode == "ycbcr":
            lr_img = lr_img.convert("YCbCr")
            hr_img = hr_img.convert("YCbCr")
        elif self.mode == "rgb":
            lr_img = lr_img.convert("RGB")
            hr_img = hr_img.convert("RGB")
        
        lr_img = TF.to_tensor(lr_img)
        hr_img = TF.to_tensor(hr_img)
        return lr_img, hr_img

def load_dataset(root_dir="./data", scale_factor=2, mode="rgb"):
    lr_dir = os.path.join(root_dir, f"X{scale_factor}/LR")
    hr_dir = os.path.join(root_dir, f"X{scale_factor}/HR")
    
    lr_img_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if is_img_file(os.path.join(lr_dir, f))])
    hr_img_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if is_img_file(os.path.join(hr_dir, f))])
    
    dataset = SRDataset(lr_img_files, hr_img_files, mode=mode)
    return dataset

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