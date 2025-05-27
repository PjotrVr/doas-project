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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                lr = TF.rotate(lr, angle=90 * k)
                hr = TF.rotate(hr, angle=90 * k)
        lr = TF.to_tensor(lr)
        hr = TF.to_tensor(hr)
        return lr, hr

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

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        """
        sign=-1 subtracts from the img, sign=+1 adds to the img.
        """
        super().__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        mean = torch.Tensor(rgb_mean)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * mean / std
        for p in self.parameters():
            p.requires_grad = False # Frozen

class ResBlock(nn.Module):
    def __init__(self, kernel_size, n_features, bias=True, residual_scaling=1.0, activation="relu"):
        super().__init__()
        padding = kernel_size // 2
        activation = activation.lower() if activation is not None else None
        layers = []
        layers.append(nn.Conv2d(n_features, n_features, kernel_size, bias=bias, padding=padding))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=False))
        elif activation == "prelu":
            layers.append(nn.PReLU(n_features))
        layers.append(nn.Conv2d(n_features, n_features, kernel_size, bias=bias, padding=padding))
        self.net = nn.Sequential(*layers)
        self.residual_scaling = residual_scaling

    def forward(self, x):
        return x + self.residual_scaling * self.net(x)

class Upsampler(nn.Module):
    def __init__(self, scale_factor, n_features, bias=True, activation=None):
        super().__init__()
        is_power_of_two = scale_factor & (scale_factor - 1) == 0
        layers = []
        activation = activation.lower() if activation is not None else None
        if is_power_of_two:
            # Assume scale_factor = 2^n, what is the n?
            # Important because we want to know how many times we have to double
            # height and width to get to the wanted scale, e.g. for scale_factor=8
            # it is 3 doublings.
            n = int(math.log(scale_factor, 2))
            for i in range(n):
                layers.append(nn.Conv2d(n_features, 4 * n_features, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(upscale_factor=2))
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=False))
                elif activation == "prelu":
                    layers.append(nn.PReLU(n_features))
        elif scale_factor == 3:
            layers.append(nn.Conv2d(n_features, 9 * n_features, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(upscale_factor=3))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=False))
            elif activation == "prelu":
                layers.append(nn.PReLU(n_features))
        else:
            raise NotImplementedError

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EDSR(nn.Module):
    def __init__(self, in_channels, n_blocks, n_features, scale_factor, activation="relu", residual_scaling=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        activation = activation.lower() if activation is not None else None
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        head_layers, body_layers, tail_layers = [], [], []
        head_layers.append(nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1))

        for i in range(n_blocks):
            body_layers.append(ResBlock(3, n_features, activation=activation, residual_scaling=residual_scaling))
        body_layers.append(nn.Conv2d(n_features, n_features, kernel_size=3, padding=1))

        tail_layers.append(Upsampler(scale_factor, n_features))
        tail_layers.append(nn.Conv2d(n_features, in_channels, kernel_size=3, padding=1))

        self.head = nn.Sequential(*head_layers)
        self.body = nn.Sequential(*body_layers)
        self.tail = nn.Sequential(*tail_layers)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        x = self.add_mean(x)
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, load_tail=True):
    state_dict = torch.load(path)
    model_dict = model.state_dict()

    if not load_tail:
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("tail")}
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)

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

def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0
    # total_samples = 0
    with torch.no_grad():
        for lr_batch, hr_batch in loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr_batch = model(lr_batch)
            sr_batch = torch.clamp(sr_batch, 0.0, 1.0)

            loss = criterion(sr_batch, hr_batch)

            total_loss += loss.item()
            total_psnr += calculate_psnr(sr_batch, hr_batch, model.scale_factor).item()
            total_ssim += ssim(sr_batch, hr_batch, data_range=1.0).item()
            # total_samples += lr_batch.shape[0]

    model.train()

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    return avg_loss, avg_psnr, avg_ssim

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

def get_div2k_dataset(data_dir="./data", scale_factor=2, patch_size=48, transform=None, mode="rgb"):
    train_hr_dir = os.path.join(data_dir, "DIV2K_train_HR")
    train_lr_dir = os.path.join(data_dir, f"DIV2K_train_LR_bicubic/X{scale_factor}")
    val_hr_dir = os.path.join(data_dir, "DIV2K_valid_HR")
    val_lr_dir = os.path.join(data_dir, f"DIV2K_valid_LR_bicubic/X{scale_factor}")

    hr_train_img_files = sorted([os.path.join(train_hr_dir, f) for f in os.listdir(train_hr_dir) if f.endswith(("png", "jpg"))])
    lr_train_img_files = sorted([os.path.join(train_lr_dir, f) for f in os.listdir(train_lr_dir) if f.endswith(("png", "jpg"))])
    hr_val_img_files = sorted([os.path.join(val_hr_dir, f) for f in os.listdir(val_hr_dir) if f.endswith(("png", "jpg"))])
    lr_val_img_files = sorted([os.path.join(val_lr_dir, f) for f in os.listdir(val_lr_dir) if f.endswith(("png", "jpg"))])

    hr_train_img_files = hr_train_img_files[:200]
    lr_train_img_files = lr_train_img_files[:200]
    hr_val_img_files = hr_val_img_files[:10]
    lr_val_img_files = lr_val_img_files[:10]
    
    train_dataset = SRDataset(lr_train_img_files, hr_train_img_files, patch_size=patch_size, scale_factor=scale_factor, mode=mode, transform=transform)
    val_dataset = SRDataset(lr_val_img_files, hr_val_img_files, patch_size=patch_size, scale_factor=scale_factor, mode=mode, transform=None)
    return train_dataset, val_dataset

seed = 0
scale_factor = 2
patch_size = 48
batch_size = 16
n_epochs = 100 # 300 # default
lr = 1e-3
n_features = 64
n_blocks = 16
transform = PairedRandomTransform(hflip=True, rot=True)
residual_scaling = 1.0
activation = "relu"
criterion = nn.L1Loss()
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "./data"

train_dataset, val_dataset = get_div2k_dataset(data_dir, scale_factor=scale_factor, patch_size=patch_size, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

seed_everything(seed)

model = EDSR(
    in_channels=3,
    n_blocks=n_blocks,
    n_features=n_features,
    scale_factor=scale_factor,
    activation=activation,
    residual_scaling=residual_scaling,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

n_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print("Number of parameters:")
print(f"\tLearnable: {n_learnable_params:,}" )
print(f"\tFrozen: {n_frozen_params:,}" )
print(f"\tTotal: {n_learnable_params+n_frozen_params:,}" )
print(f"Starting training on {device}...")

seed_everything(seed)
best_val_loss = float("inf")
history = {
    "train_loss": [],
    "train_psnr": [],
    "train_ssim": [],
    "val_loss": [],
    "val_psnr": [],
    "val_ssim": [],
}

for epoch in range(n_epochs):
    train_loss, train_psnr, train_ssim = train_one_epoch(model, criterion, optimizer, train_loader, device)
    val_loss, val_psnr, val_ssim = evaluate(model, criterion, val_loader, device)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_psnr"].append(train_psnr)
    history["train_ssim"].append(train_ssim)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim)

    print(f"Epoch {epoch+1}/{n_epochs}:")
    print(f"\tTrain: Loss={train_loss:.4f}, PSNR={train_psnr:.3f}dB, SSIM={train_ssim:.5f}")
    print(f"\tVal: Loss={val_loss:.4f}, PSNR={val_psnr:.3f}dB, SSIM={val_ssim:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("\tFound best model! Saving...")
        save_model(model, f"best_edsr_x{scale_factor}_best.pt")