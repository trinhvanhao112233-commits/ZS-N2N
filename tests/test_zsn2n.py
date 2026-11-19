"""Kiem tra tu dong cac thanh phan Phan 2 ZS-N2N."""

import unittest

import torch
import torch.nn.functional as F

from src.input_utils import set_reproducible
from src.losses import loss_func
from src.model import network
from src.pair_downsampler import pair_downsampler


class TestZSN2N(unittest.TestCase):
    def setUp(self) -> None:
        set_reproducible(2023)
        self.model = network(n_chan=3)
        self.image = torch.rand(1, 3, 16, 20)

    def test_01_parameter_count(self) -> None:
        count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(count, 22_275)

    def test_02_network_preserves_shape(self) -> None:
        self.assertEqual(self.model(self.image).shape, self.image.shape)

    def test_03_pair_downsampler_kernels_and_shape(self) -> None:
        block = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        d1, d2 = pair_downsampler(block)
        self.assertTrue(torch.equal(d1, torch.tensor([[[[2.5]]]])))
        self.assertTrue(torch.equal(d2, torch.tensor([[[[2.5]]]])))

        rgb = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
        rgb_d1, rgb_d2 = pair_downsampler(rgb)
        self.assertEqual(rgb_d1.shape, (1, 3, 2, 3))
        self.assertEqual(rgb_d2.shape, (1, 3, 2, 3))
        expected_d1 = 0.5 * (rgb[:, :, 0, 1] + rgb[:, :, 1, 0])
        expected_d2 = 0.5 * (rgb[:, :, 0, 0] + rgb[:, :, 1, 1])
        self.assertTrue(torch.equal(rgb_d1[:, :, 0, 0], expected_d1))
        self.assertTrue(torch.equal(rgb_d2[:, :, 0, 0], expected_d2))

    def test_04_loss_components_match_formulas(self) -> None:
        total, residual, consistency = loss_func(self.image, self.model)
        d1, d2 = pair_downsampler(self.image)
        p1 = d1 - self.model(d1)
        p2 = d2 - self.model(d2)
        expected_residual = 0.5 * (F.mse_loss(p1, d2) + F.mse_loss(p2, d1))
        denoised = self.image - self.model(self.image)
        a1, a2 = pair_downsampler(denoised)
        expected_consistency = 0.5 * (
            F.mse_loss(p1, a1) + F.mse_loss(p2, a2)
        )
        self.assertTrue(torch.allclose(residual, expected_residual, atol=1e-8))
        self.assertTrue(torch.allclose(consistency, expected_consistency, atol=1e-8))
        self.assertTrue(torch.allclose(total, residual + consistency, atol=1e-8))

    def test_05_total_is_sum_of_components(self) -> None:
        total, residual, consistency = loss_func(self.image, self.model)
        error = abs(total.item() - residual.item() - consistency.item())
        self.assertLessEqual(error, 1e-7)

    def test_06_backpropagation_computes_gradients(self) -> None:
        total, _, _ = loss_func(self.image, self.model)
        total.backward()
        gradients = [parameter.grad for parameter in self.model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

