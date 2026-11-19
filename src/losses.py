"""Residual Loss va Consistency Loss cua ZS-N2N - pham vi Nguoi 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pair_downsampler import pair_downsampler


def loss_func(
    noisy_img: torch.Tensor, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tra ve (Total Loss, Residual Loss, Consistency Loss)."""
    noisy1, noisy2 = pair_downsampler(noisy_img)

    # Cong thuc (4): hoc phan du doi xung giua hai anh con.
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (
        F.mse_loss(pred1, noisy2) + F.mse_loss(pred2, noisy1)
    )

    # Cong thuc (6): tach-truoc/khu-nhieu-sau nhat quan voi
    # khu-nhieu-truoc/tach-sau. Ca ba lan goi dung chung model.
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (
        F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2)
    )
    return loss_res + loss_cons, loss_res, loss_cons

