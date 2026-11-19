"""Demo truc quan Phan 1: Nap anh, them nhieu va tach cap anh con (ZS-N2N).

Phu hop voi pham vi Nguoi 1:
- §10.1 - 10.3: Nap anh (sach hoac tu tao), them nhieu Gaussian / Poisson.
- §10.5: Ap dung bo tach cap anh pair_downsampler.
- §10.6: Hien thi va luu truc quan anh nhieu va 2 anh con D1, D2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.input_utils import (
    add_noise,
    load_rgb_image,
    make_synthetic_clean_image,
    save_tensor_image,
    set_reproducible,
)
from src.pair_downsampler import pair_downsampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo Phan 1: Tao nhieu va tach cap anh con (Pair Downsampler)"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Duong dan anh RGB sach tuy chon (mac dinh: tao anh tong hop)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Kich thuoc anh tong hop neu khong nap anh tu ngoai (mac dinh: 256)",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gauss",
        choices=["gauss", "poiss"],
        help="Loai nhieu can them: 'gauss' (Gaussian) hoac 'poiss' (Poisson)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=25.0,
        help="Muc nhieu: do lech chuan sigma voi Gauss (vi du: 25.0); lambda voi Poisson (vi du: 30.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed de ket qua thuc nghiem co the tai lap",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Thu muc luu ket qua anh va bieu do",
    )
    return parser


def tensor_to_numpy(tensor: torch.Tensor):
    """Chuyen tensor [1, C, H, W] hoac [C, H, W] sang numpy [H, W, C] de hien thi voi matplotlib."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def main() -> None:
    args = build_parser().parse_args()
    set_reproducible(args.seed)

    plot_dir = args.output_dir / "plots"
    image_dir = args.output_dir / "images"
    log_dir = args.output_dir / "logs"
    for directory in (plot_dir, image_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cropped = False
    clean_available = False
    if args.image is not None:
        if not args.image.exists():
            raise SystemExit(f"Loi: Tep anh khong ton tai: {args.image}")
        clean_img, cropped = load_rgb_image(args.image)
        source_name = f"Anh nạp từ tệp: {args.image.name}"
        clean_available = True
    else:
        clean_img = make_synthetic_clean_image(size=args.size)
        source_name = f"Ảnh tổng hợp cấu trúc ({args.size}x{args.size} RGB)"
        clean_available = True

    # §10.3: Thêm nhiễu vào ảnh sạch
    noisy_img = add_noise(
        clean_img, noise_type=args.noise_type, noise_level=args.noise_level
    )

    # §10.5: Tách cặp ảnh con
    d1_img, d2_img = pair_downsampler(noisy_img)

    lines: list[str] = []

    def report(text: str = "") -> None:
        print(text)
        lines.append(text)

    report("============================================================")
    report("         DEMO PHẦN 1: BỘ TÁCH CẶP ẢNH CON (ZS-N2N)          ")
    report("============================================================")
    report(f"Nguồn ảnh đầu vào : {source_name}")
    if cropped:
        report("Lưu ý             : Đã cắt 1 pixel biên để kích thước H và W chẵn.")
    report(f"Kích thước ảnh sạch: {list(clean_img.shape)}")
    report(f"Loại nhiễu        : {args.noise_type.upper()}")
    report(f"Mức nhiễu (level) : {args.noise_level}")
    report(f"Random Seed       : {args.seed}")
    report("------------------------------------------------------------")
    report("1. KÍCH THƯỚC VÀ THÔNG SỐ CÁC TENSOR:")
    report(f"- Ảnh nhiễu y     : shape={list(noisy_img.shape)}, min={noisy_img.min():.4f}, max={noisy_img.max():.4f}")
    report(f"- Ảnh con D1(y)   : shape={list(d1_img.shape)}, min={d1_img.min():.4f}, max={d1_img.max():.4f}")
    report(f"- Ảnh con D2(y)   : shape={list(d2_img.shape)}, min={d2_img.min():.4f}, max={d2_img.max():.4f}")

    # Đánh giá tương quan và độ lệch giữa 2 ảnh con
    mse_d1_d2 = torch.mean((d1_img - d2_img) ** 2).item()
    report("------------------------------------------------------------")
    report("2. PHÂN TÍCH TÍNH CHẤT HAI ẢNH CON D1(y) VÀ D2(y):")
    report(f"- MSE giữa D1 và D2: {mse_d1_d2:.6f}")
    report("  -> Nhận xét: D1 và D2 có chung tín hiệu nền nhưng hoa văn nhiễu độc lập.")
    report("  -> Mỗi ảnh con giảm một nửa chiều cao và chiều rộng (H/2, W/2).")
    report("  -> Phép lấy trung bình 2 pixel trên đường chéo còn hỗ trợ giảm nhẹ phương sai nhiễu.")
    report("============================================================")

    # §10.6: Vẽ và hiển thị trực quan các ảnh
    # Bố cục 4 ảnh: [Ảnh sạch, Ảnh nhiễu y, Ảnh con D1, Ảnh con D2]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.0))

    axes[0].imshow(tensor_to_numpy(clean_img))
    axes[0].set_title(f"Ảnh sạch x\n({clean_img.shape[2]}x{clean_img.shape[3]})", fontsize=11, pad=8)
    axes[0].axis("off")

    axes[1].imshow(tensor_to_numpy(noisy_img))
    axes[1].set_title(
        f"Ảnh nhiễu y ({args.noise_type.upper()} {args.noise_level})\n({noisy_img.shape[2]}x{noisy_img.shape[3]})",
        fontsize=11,
        pad=8,
    )
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(d1_img))
    axes[2].set_title(
        f"Ảnh con D1(y) [k1: chéo phụ]\n({d1_img.shape[2]}x{d1_img.shape[3]})",
        fontsize=11,
        pad=8,
    )
    axes[2].axis("off")

    axes[3].imshow(tensor_to_numpy(d2_img))
    axes[3].set_title(
        f"Ảnh con D2(y) [k2: chéo chính]\n({d2_img.shape[2]}x{d2_img.shape[3]})",
        fontsize=11,
        pad=8,
    )
    axes[3].axis("off")

    fig.suptitle(
        "ZS-N2N: Minh họa bộ tách cặp ảnh con (Pair Downsampler)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plot_file = plot_dir / f"phan1_downsampling_{args.noise_type}.png"
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close()

    # Lưu các file ảnh riêng biệt phục vụ báo cáo
    save_tensor_image(clean_img, image_dir / "phan1_clean.png")
    save_tensor_image(noisy_img, image_dir / f"phan1_noisy_{args.noise_type}.png")
    save_tensor_image(d1_img, image_dir / "phan1_d1.png")
    save_tensor_image(d2_img, image_dir / "phan1_d2.png")

    log_file = log_dir / "ket_qua_phan1.txt"
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[OK] Đã lưu hình ảnh trực quan vào: {plot_file}")
    print(f"[OK] Đã lưu các ảnh thành phần vào : {image_dir}")
    print(f"[OK] Đã lưu nhật ký log vào       : {log_file}")


if __name__ == "__main__":
    main()
