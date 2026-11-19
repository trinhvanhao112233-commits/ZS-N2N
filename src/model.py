"""Kien truc CNN nho cua ZS-N2N - pham vi Nguoi 2."""

import torch
import torch.nn as nn


class network(nn.Module):
    """CNN uoc luong nhieu: 3 -> 48 -> 48 -> 3 cho anh RGB."""

    def __init__(self, n_chan: int = 3, chan_embed: int = 48) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def parameter_counts(model: network) -> dict[str, int]:
    """Dem tham so theo tung lop de doi chieu bao cao."""
    return {
        "Conv1": sum(parameter.numel() for parameter in model.conv1.parameters()),
        "Conv2": sum(parameter.numel() for parameter in model.conv2.parameters()),
        "Conv3": sum(parameter.numel() for parameter in model.conv3.parameters()),
    }

