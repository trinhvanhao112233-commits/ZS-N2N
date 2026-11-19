"""Doc anh va tao anh nhieu - pham vi Nguoi 1."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image


def set_reproducible(seed: int) -> None:
    """Co dinh seed ngau nhien cho ket qua co the tai lap."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_clean_image(size: int = 128) -> torch.Tensor:
    """Tao mot anh RGB sach nhan tao co cau truc (hinh hoc va chuyen sac)."""
    coord = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    base = 0.25 + 0.42 * xx + 0.16 * yy
    blob = 0.18 * torch.exp(-((xx - 0.65) ** 2 + (yy - 0.42) ** 2) / 0.022)
    clean = torch.stack(
        (base + blob, base * 0.88 + blob * 0.7, base * 0.65 + blob), dim=0
    ).unsqueeze(0)
    return clean.clamp(0.0, 1.0)


def add_noise(
    clean: torch.Tensor,
    noise_type: str = "gauss",
    noise_level: float = 25.0,
) -> torch.Tensor:
    """Them nhieu Gaussian hoac Poisson vao anh sach theo muc §10.3.
    
    Args:
        clean: Tensor anh sach thang [0, 1], dang [N, C, H, W].
        noise_type: 'gauss' hoac 'poiss'.
        noise_level:
            - Voi Gaussian: do lech chuan sigma tren thang 0-255 (vi du: 25.0 -> sigma = 25/255).
            - Voi Poisson: he so ty le lambda (vi du: 30.0; lambda cang nho nhieu cang nang).
    """
    clean_clamped = clean.clamp(0.0, 1.0)
    if noise_type.lower() == "gauss":
        sigma = noise_level / 255.0
        noisy = clean_clamped + torch.randn_like(clean_clamped) * sigma
    elif noise_type.lower() in ("poiss", "poisson"):
        # Theo §10.3: noisy = torch.poisson(noise_level * x) / noise_level
        noisy = torch.poisson(clean_clamped * noise_level) / noise_level
    else:
        raise ValueError(f"Khong ho tro loai nhieu: '{noise_type}'. Chi dung 'gauss' hoac 'poiss'.")
    return noisy.clamp(0.0, 1.0)


def make_noisy_image(
    size: int = 128,
    sigma: float | None = None,
    noise_type: str = "gauss",
    noise_level: float = 25.0,
) -> torch.Tensor:
    """Tao anh RGB va them nhieu (giu tuong thich cho cac module demo truoc)."""
    clean = make_synthetic_clean_image(size=size)
    if sigma is not None:
        noise_level = sigma * 255.0
    return add_noise(clean, noise_type=noise_type, noise_level=noise_level)


def load_rgb_image(path: Path) -> tuple[torch.Tensor, bool]:
    """Doc anh RGB; cat 1 pixel o canh phai/duoi neu chieu dai hoac rong bi le."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        even_width, even_height = width - width % 2, height - height % 2
        if even_width < 2 or even_height < 2:
            raise ValueError("Anh phai co chieu cao va chieu rong it nhat bang 2.")
        cropped = (even_width, even_height) != (width, height)
        if cropped:
            image = image.crop((0, 0, even_width, even_height))
        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = raw.reshape(even_height, even_width, 3).permute(2, 0, 1)
        return tensor.unsqueeze(0).float() / 255.0, cropped


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Luu tensor anh thang [0, 1] [1, C, H, W] thanh file anh RGB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255
    ).round().to(torch.uint8).numpy()
    Image.fromarray(array, mode="RGB").save(path)
