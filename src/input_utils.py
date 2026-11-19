"""Doc anh va tao anh nhieu - pham vi Nguoi 1."""

import random
from pathlib import Path

import torch
from PIL import Image


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_noisy_image(size: int = 128, sigma: float = 25 / 255) -> torch.Tensor:
    """Tao anh RGB co cau truc va them nhieu Gaussian de demo ngay."""
    coord = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    base = 0.25 + 0.42 * xx + 0.16 * yy
    blob = 0.18 * torch.exp(-((xx - 0.65) ** 2 + (yy - 0.42) ** 2) / 0.022)
    clean = torch.stack(
        (base + blob, base * 0.88 + blob * 0.7, base * 0.65 + blob), dim=0
    ).unsqueeze(0)
    # Anh clean chi dung de tao anh nhieu, khong tham gia loss.
    return (clean.clamp(0, 1) + torch.randn_like(clean) * sigma).clamp(0, 1)


def load_rgb_image(path: Path) -> tuple[torch.Tensor, bool]:
    """Doc anh RGB; cat 1 pixel o canh phai/duoi neu kich thuoc le."""
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
    array = (
        tensor.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255
    ).round().to(torch.uint8).numpy()
    Image.fromarray(array, mode="RGB").save(path)

