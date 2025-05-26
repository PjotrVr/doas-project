import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def nearest_neighbor(img, scale_factor=2):
    """
    Assumes `img` has shape (H, W) or (H, W, C).
    """
    if img.ndim == 2:
        img = img[:, :, None]

    H, W = img.shape[:2]
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    row_idx = (np.arange(new_H) / scale_factor).astype(np.int32)
    col_idx = (np.arange(new_W) / scale_factor).astype(np.int32)

    row_idx = np.clip(row_idx, 0, H - 1)
    col_idx = np.clip(col_idx, 0, W - 1)
    return img[row_idx[:, None], col_idx]

def bilinear(img, scale_factor=2):
    """
    Assumes `img` has shape (H, W) or (H, W, C).
    """
    if img.ndim == 2:
        img = img[:, :, None]

    H, W, channels = img.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    result = np.zeros((new_H, new_W, channels), dtype=np.float32)
    for i in range(new_H):
        for j in range(new_W):
            src_y = i / scale_factor
            src_x = j / scale_factor

            y0 = int(np.floor(src_y))
            x0 = int(np.floor(src_x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)

            dy = src_y - y0
            dx = src_x - x0

            result[i, j] = (1 - dy) * (1 - dx) * img[y0, x0] \
                            + (1 - dy) * dx * img[y0, x1] \
                            + dy * (1 - dx) * img[y1, x0] \
                            + dy * dx * img[y1, x1]

    result = np.clip(result, 0.0, 1.0)
    return result

def keys_cubic_kernel(t, alpha=-0.5):
    t = np.abs(t)
    t2 = t**2
    t3 = t2 * t

    result = np.zeros_like(t)

    mask1 = (t <= 1)
    result[mask1] = ((alpha + 2) * t3[mask1] -
                     (alpha + 3) * t2[mask1] + 1)

    mask2 = (t > 1) & (t < 2)
    result[mask2] = (alpha * t3[mask2] -
                     5 * alpha * t2[mask2] +
                     8 * alpha * t[mask2] - 4 * alpha)

    return result

def bicubic(img, scale_factor=2.0, alpha=-0.5):
    """
    Assumes `img` has shape (H, W) or (H, W, C).
    """
    if img.ndim == 2:
        img = img[:, :, None]

    H, W, C = img.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    result = np.zeros((new_H, new_W, C), dtype=np.float32)

    for i in range(new_H):
        for j in range(new_W):
            # Map back to original image
            y = i / scale_factor
            x = j / scale_factor

            y_int = int(np.floor(y))
            x_int = int(np.floor(x))

            dy = y - y_int
            dx = x - x_int

            patch = np.zeros((4, 4, C), dtype=np.float32)

            for m in range(-1, 3):
                for n in range(-1, 3):
                    yy = np.clip(y_int + m, 0, H - 1)
                    xx = np.clip(x_int + n, 0, W - 1)
                    patch[m + 1, n + 1] = img[yy, xx]

            # Compute weights
            wy = np.array([keys_cubic_kernel(m - dy, alpha) for m in range(-1, 3)])
            wx = np.array([keys_cubic_kernel(n - dx, alpha) for n in range(-1, 3)])

            # Interpolate
            for c in range(C):
                result[i, j, c] = np.dot(wy, np.dot(patch[:, :, c], wx))

    result = np.clip(result, 0.0, 1.0)
    return result

# Metrics
def mse(original_img, recreated_img):
    return np.mean((original_img - recreated_img) ** 2)

def rmse(original_img, recreated_img):
    return np.sqrt(mse(original_img, recreated_img))

def psnr(original_img, recreated_img, max_pixel_value=1.0):
    rmse_value = rmse(original_img, recreated_img)
    if rmse_value == 0:
        rmse_value = float("inf")
    return 20 * np.log10(max_pixel_value / rmse_value)