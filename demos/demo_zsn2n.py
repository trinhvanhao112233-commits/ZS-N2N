"""Demo CPU cho CNN va hai thanh phan loss cua ZS-N2N."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.input_utils import (
    load_rgb_image,
    make_noisy_image,
    save_tensor_image,
    set_reproducible,
)
from src.losses import loss_func
from src.model import network, parameter_counts
from src.pair_downsampler import pair_downsampler


def fmt_loss(name: str, value: float) -> str:
    return f"{name:<24}: {value:.9f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo Phan 2 ZS-N2N tren CPU")
    parser.add_argument("--image", type=Path, help="Duong dan anh RGB tuy chon")
    parser.add_argument("--steps", type=int, default=20, help="So vong lap (mac dinh: 20)")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate Adam")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 0:
        raise SystemExit("Loi: --steps phai >= 0.")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise SystemExit("Loi: --lr phai la so duong huu han.")

    set_reproducible(args.seed)
    torch.set_num_threads(2)
    image_dir = args.output_dir / "images"
    plot_dir = args.output_dir / "plots"
    log_dir = args.output_dir / "logs"
    for directory in (image_dir, plot_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cropped = False
    if args.image is None:
        noisy_img = make_noisy_image()
        source = "Anh nhieu tong hop 128x128 RGB, sigma=25/255"
    else:
        noisy_img, cropped = load_rgb_image(args.image)
        source = f"Anh thuc: {args.image}"

    model = network(n_chan=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    counts = parameter_counts(model)
    total_params = sum(counts.values())
    noisy1, noisy2 = pair_downsampler(noisy_img)

    with torch.no_grad():
        total_before, res_before, cons_before = loss_func(noisy_img, model)

    lines: list[str] = []

    def report(text: str = "") -> None:
        print(text)
        lines.append(text)

    report("=== ZS-N2N: DEMO PHAN 2 (CPU) ===")
    report(f"Nguon anh: {source}")
    if cropped:
        report("Luu y: da cat canh phai/duoi 1 pixel de H va W la so chan.")
    report(f"Seed: {args.seed} | Adam lr: {args.lr} | So vong lap: {args.steps}")
    report("--- SO THAM SO CNN ---")
    for name, count in counts.items():
        report(f"{name:<8}: {count:>6,}")
    report(f"Tong    : {total_params:>6,} (yeu cau: 22,275)")
    report("--- KICH THUOC TENSOR ---")
    report(f"Anh y   : {tuple(noisy_img.shape)}")
    report(f"D1(y)   : {tuple(noisy1.shape)}")
    report(f"D2(y)   : {tuple(noisy2.shape)}")
    report("--- TRUOC HUAN LUYEN ---")
    report(fmt_loss("Residual Loss", res_before.item()))
    report(fmt_loss("Consistency Loss", cons_before.item()))
    report(fmt_loss("Total Loss", total_before.item()))

    history: list[float] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = loss_func(noisy_img, model)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            evaluated_total, _, _ = loss_func(noisy_img, model)
        history.append(evaluated_total.item())
        if step in {1, 5, 10, args.steps}:
            report(f"Vong {step:>3}/{args.steps}: Total Loss = {history[-1]:.9f}")

    with torch.no_grad():
        total_after, res_after, cons_after = loss_func(noisy_img, model)
        denoised = (noisy_img - model(noisy_img)).clamp(0, 1)
    sum_error = abs(total_after.item() - (res_after.item() + cons_after.item()))
    formula_ok = sum_error <= 1e-7
    report("--- SAU HUAN LUYEN ---")
    report(fmt_loss("Residual Loss", res_after.item()))
    report(fmt_loss("Consistency Loss", cons_after.item()))
    report(fmt_loss("Total Loss", total_after.item()))
    report(fmt_loss("Sai lech Total-(Res+Cons)", sum_error))
    report(f"Kiem chung cong thuc Loss: {'DAT' if formula_ok else 'KHONG DAT'}")
    report("Ghi chu: Loss khong duoc dung de tu khang dinh chat luong khu nhieu.")

    result_path = log_dir / "ket_qua_zsn2n.txt"
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt.figure(figsize=(7, 4))
    if history:
        plt.plot(range(1, args.steps + 1), history, color="tab:blue", linewidth=1.8)
    else:
        plt.scatter([0], [total_before.item()], color="tab:blue")
    plt.xlabel("Vong lap")
    plt.ylabel("Total Loss")
    plt.title("ZS-N2N - Total Loss theo vong lap")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    chart_path = plot_dir / "total_loss.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    if args.image is not None:
        save_tensor_image(noisy_img, image_dir / "anh_dau_vao.png")
        save_tensor_image(denoised, image_dir / "anh_sau_khu_nhieu.png")

    print(f"Da luu ket qua: {result_path}")
    print(f"Da luu bieu do: {chart_path}")


if __name__ == "__main__":
    main()

