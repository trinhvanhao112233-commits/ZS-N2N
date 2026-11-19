"""Pair downsampler toi thieu de tich hop; Nguoi 1 tiep tuc quan ly."""

import torch
import torch.nn.functional as F


def pair_downsampler(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tao D1=(b+c)/2 va D2=(a+d)/2 tren tung khoi 2x2, tung kenh."""
    if img.ndim != 4:
        raise ValueError("Anh dau vao phai co dang [N, C, H, W].")
    channels = img.shape[1]
    kernel_d1 = img.new_tensor([[[[0.0, 0.5], [0.5, 0.0]]]])
    kernel_d2 = img.new_tensor([[[[0.5, 0.0], [0.0, 0.5]]]])
    kernel_d1 = kernel_d1.repeat(channels, 1, 1, 1)
    kernel_d2 = kernel_d2.repeat(channels, 1, 1, 1)
    d1 = F.conv2d(img, kernel_d1, stride=2, groups=channels)
    d2 = F.conv2d(img, kernel_d2, stride=2, groups=channels)
    return d1, d2

