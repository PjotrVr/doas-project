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

class SRCNN(nn.Module):
    def __init__(self, in_channels=1, n1=64, n2=32, f1=9, f2=1, f3=5, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        padding1 = (f1 - 1) // 2
        padding2 = (f2 - 1) // 2
        padding3 = (f3 - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=f1, padding=padding1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n1, n2, kernel_size=f2, padding=padding2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n2, in_channels, kernel_size=f3, padding=padding3)
        )

    def forward(self, x, eps=1e-9):
        interpolated_x = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        interpolated_x = interpolated_x.clamp(min=eps, max=1-eps)
        return self.net(interpolated_x).clamp(min=eps, max=1-eps)

class DIV2KPatchDataset(data.Dataset):
    def __init__(self, lr_paths, hr_paths, patch_size=96, scale_factor=2):
        assert len(lr_paths) == len(hr_paths), "Mismatched number of images"
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.lr_patch_size = patch_size // scale_factor

    def __len__(self):
        return len(self.lr_paths)

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
    train_hr_dir = "./data/DIV2K_train_HR"
    train_lr_dir = f"./data/DIV2K_train_LR_bicubic/X{scale_factor}"
    val_hr_dir = "./data/DIV2K_valid_HR"
    val_lr_dir = f"./data/DIV2K_valid_LR_bicubic/X{scale_factor}"

    hr_train_img_files = sorted([os.path.join(train_hr_dir, f) for f in os.listdir(train_hr_dir) if f.endswith(("png", "jpg"))])
    lr_train_img_files = sorted([os.path.join(train_lr_dir, f) for f in os.listdir(train_lr_dir) if f.endswith(("png", "jpg"))])
    hr_val_img_files = sorted([os.path.join(val_hr_dir, f) for f in os.listdir(val_hr_dir) if f.endswith(("png", "jpg"))])
    lr_val_img_files = sorted([os.path.join(val_lr_dir, f) for f in os.listdir(val_lr_dir) if f.endswith(("png", "jpg"))])

    train_dataset = DIV2KPatchDataset(lr_train_img_files, hr_train_img_files, patch_size=patch_size, scale_factor=scale_factor)
    val_dataset = DIV2KPatchDataset(lr_val_img_files, hr_val_img_files, patch_size=patch_size, scale_factor=scale_factor)
    return train_dataset, val_dataset

def calculate_psnr(preds, target, max_val=1.0):
    mse = F.mse_loss(preds, target, reduction="mean")
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss, total_psnr = 0.0, 0.0
    # total_samples = 0
    for lr_batch, hr_batch in loader:
        lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
        sr_batch = model(lr_batch)
        loss = F.mse_loss(sr_batch, hr_batch, reduction="mean")
        psnr = calculate_psnr(sr_batch, hr_batch)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_psnr += psnr.item()
        # total_samples += lr_batch.shape[0]

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    return avg_loss, avg_psnr

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_psnr = 0.0, 0.0
    # total_samples = 0
    with torch.no_grad():
        for lr_batch, hr_batch in loader:
            lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
            sr_batch = model(lr_batch)
            
            loss = F.mse_loss(sr_batch, hr_batch, reduction="mean")
            psnr = calculate_psnr(sr_batch, hr_batch)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            # total_samples += lr_batch.shape[0]

    model.train()

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    return avg_loss, avg_psnr
    
# Hyperparameters
scale_factor = 4
batch_size = 64
lr = 1e-3
patch_size = 256
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
train_dataset, val_dataset = get_div2k_dataset(scale_factor=scale_factor, patch_size=patch_size)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

torch.manual_seed(0)
model = SRCNN(in_channels=3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Starting training on {device}...")

torch.manual_seed(0)
best_val_loss = float("inf")
history = {
    "train_loss": [],
    "train_psnr": [],
    "val_loss": [],
    "val_psnr": [],
}

for epoch in range(n_epochs):
    train_loss, train_psnr = train_one_epoch(model, optimizer, train_loader, device)
    val_loss, val_psnr = evaluate(model, val_loader, device)

    history["train_loss"].append(train_loss)
    history["train_psnr"].append(train_psnr)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)

    print(f"Epoch {epoch+1}/{n_epochs}")
    print(f"\tTrain: Loss={train_loss:.4f}, PSNR={train_psnr:.3f}dB")
    print(f"\tVal: Loss={val_loss:.4f}, PSNR={val_psnr:.3f}dB")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print("Found best model! Saving...")
        torch.save(model.state_dict(), "best_model.pth")
        
print("Finished training")