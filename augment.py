import os
import random

from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models.utils import is_img_file

# Taken from: https://github.com/wangzhesun/super_resolution
# In paper they use hr_patch_size
def random_crop(lr_img, hr_img, lr_patch_size=None, hr_patch_size=None, scale_factor=2):
    assert (lr_patch_size is not None) ^ (hr_patch_size is not None), "Exactly one of lr_patch_size or hr_patch_size must be provided."
    if hr_patch_size is not None:
        lr_patch_size = hr_patch_size // scale_factor
    else:
        hr_patch_size = lr_patch_size * scale_factor

    h, w = lr_img.shape[1:3]

    lr_width = random.randint(0, w - lr_patch_size)
    lr_height = random.randint(0, h - lr_patch_size)
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

def augment_img(lr_img_file, hr_img_file, output_dir, n_augmentations, scale_factor=2, lr_patch_size=None, hr_patch_size=None):
    lr_img = TF.to_tensor(Image.open(lr_img_file).convert("RGB"))
    hr_img = TF.to_tensor(Image.open(hr_img_file).convert("RGB"))

    for i in range(n_augmentations):
        lr_cropped_img, hr_cropped_img = random_crop(
            lr_img,
            hr_img, 
            lr_patch_size=lr_patch_size,
            hr_patch_size=hr_patch_size,
            scale_factor=scale_factor
        )
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
            
def augment_dir(data_dir, output_dir, n_augmentations=5, lr_patch_size=None, hr_patch_size=None, scale_factor=2):
    lr_dir = os.path.join(data_dir, f"X{scale_factor}/LR")
    hr_dir = os.path.join(data_dir, f"X{scale_factor}/HR")
    lr_img_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if is_img_file(os.path.join(lr_dir, f))])
    hr_img_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if is_img_file(os.path.join(hr_dir, f))])
    
    os.makedirs(os.path.join(output_dir, f"X{scale_factor}/LR"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"X{scale_factor}/HR"), exist_ok=True)
    
    for i, (lr_img_file, hr_img_file) in enumerate(zip(lr_img_files, hr_img_files)):
        print(f"Augmenting image {i + 1}...")
        augment_img(
            lr_img_file=lr_img_file,
            hr_img_file=hr_img_file,
            output_dir=output_dir,
            n_augmentations=n_augmentations, 
            lr_patch_size=lr_patch_size,
            hr_patch_size=hr_patch_size,
            scale_factor=scale_factor
        )

if __name__ == "__main__":
    # for dataset in ["Urban100", "Set5", "Set14", "BSD100", "DIV2K-val", "DIV2K-train"]:
    for dataset in ["Set5"]:
        for scale_factor in [2, 3, 4]:
            random.seed(0)
            augment_dir(
                data_dir=f"./data/{dataset}",
                output_dir=f"./data-augmented4/{dataset}",
                n_augmentations=20,
                scale_factor=scale_factor,
                hr_patch_size=192
                # lr_patch_size=48
            )