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
    def __init__(self, kernel_size, n_features, bias=True, res_scale=1.0, activation="relu"):
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
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.net(x)

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
    def __init__(self, in_channels, n_blocks, n_features, scale_factor, activation="relu", res_scale=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        activation = activation.lower() if activation is not None else None
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        head_layers, body_layers, tail_layers = [], [], []
        head_layers.append(nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1))

        for i in range(n_blocks):
            body_layers.append(ResBlock(3, n_features, activation=activation, res_scale=res_scale))
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

def load_model(model, path, load_tail=True, device=torch.device("cpu")):
    state_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()

    if not load_tail:
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("tail")}
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)
