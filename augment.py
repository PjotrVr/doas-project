import os
import random

from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Taken from: https://github.com/wangzhesun/super_resolution
# In paper they use hr_patch_size
def random_crop(lr_img, hr_img, lr_patch_size=48, scale_factor=2):
    hr_patch_size = lr_patch_size * scale_factor
    h, w = lr_img.shape[1:3]

    lr_width = random.randint(0, w - lr_patch_size + 1)
    lr_height = random.randint(0, h - lr_patch_size + 1)
    hr_width = lr_width * scale_factor
    hr_height = lr_height * scale_factor

    lr_img_cropped = lr_img[:, lr_height:lr_height + lr_patch_size, lr_width:lr_width + lr_patch_size]
    hr_img_cropped = hr_img[:, hr_height:hr_height + hr_patch_size, hr_width:hr_width + hr_patch_size]
    return lr_img_cropped, hr_img_cropped

def random_flip(lr_img, hr_img):
    if random.random() > 0.5:
        lr_img = torch.flip(lr_img, (2,))
        hr_img = torch.flip(hr_img, (2,))
    return lr_img, hr_img    

def random_rotate(lr_img, hr_img):
    random_val = random.randint(0, 3)
    return torch.rot90(lr_img, random_val, (1, 2)), torch.rot90(hr_img, random_val, (1, 2))

def augment_img(lr_img_file, hr_img_file, output_dir, n_augmentations, lr_patch_size, scale_factor):
    lr_img = TF.to_tensor(Image.open(lr_img_file).convert("RGB"))
    hr_img = TF.to_tensor(Image.open(hr_img_file).convert("RGB"))

    for i in range(n_augmentations):
        lr_cropped_img, hr_cropped_img = random_crop(lr_img, hr_img, lr_patch_size=lr_patch_size, scale_factor=scale_factor)
        if random.random() < 0.5:
            lr_cropped_img, hr_cropped_img = random_flip(lr_cropped_img, hr_cropped_img)
        if random.random() < 0.5:
            lr_cropped_img, hr_cropped_img = random_rotate(lr_cropped_img, hr_cropped_img)
    
        lr_filename = os.path.splitext(os.path.basename(lr_img_file))[0]
        hr_filename = os.path.splitext(os.path.basename(hr_img_file))[0]

        lr_out = f"{output_dir}/X{scale_factor}/LR/{lr_filename}_{i + 1}.png"
        hr_out = f"{output_dir}/X{scale_factor}/HR/{hr_filename}_{i + 1}.png"

        TF.to_pil_image(lr_cropped_img).save(lr_out)
        TF.to_pil_image(hr_cropped_img).save(hr_out)
            
def augment_dir(data_dir, output_dir, n_augmentations=5, lr_patch_size=48, scale_factor=2):
    lr_files = sorted([f for f in os.listdir(f"{data_dir}/X{scale_factor}/LR") if f.endswith((".png", ".jpg"))])
    hr_files = sorted([f for f in os.listdir(f"{data_dir}/X{scale_factor}/HR") if f.endswith((".png", ".jpg"))])
    os.makedirs(f"{output_dir}/X{scale_factor}/LR", exist_ok=True)
    os.makedirs(f"{output_dir}/X{scale_factor}/HR", exist_ok=True)
    
    for i in range(len(lr_files)):
        print(f"Augmenting image {i + 1}...")
        augment_img(
            os.path.join(f"{data_dir}/X{scale_factor}/LR", lr_files[i]),
            os.path.join(f"{data_dir}/X{scale_factor}/HR", hr_files[i]),
            output_dir,
            n_augmentations=n_augmentations, 
            lr_patch_size=lr_patch_size, 
            scale_factor=scale_factor
        )

if __name__ == "__main__":
    for dataset in ["DIV2K-train", "DIV2K-val", "Urban100", "Set5", "Set14", "BSD100"]:
        random.seed(0)
        augment_dir(
            data_dir=f"./data/{dataset}",
            output_dir=f"./data-augmented/{dataset}",
            n_augmentations=20,
            scale_factor=2
        )