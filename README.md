# Zero-Shot Noise2Noise

Du an nhom minh hoa bai bao **Zero-Shot Noise2Noise: Efficient Image Denoising without any Data** (CVPR 2023).

Nhanh `main` hien chua ban tich hop chay duoc cua Phan 2: mang CNN 22.275 tham so, pair downsampler, Residual Loss, Consistency Loss, demo CPU va kiem thu tu dong. Demo ngan khong dai dien cho thiet lap thuc nghiem day du cua bai bao.

## Cau truc va phan cong

```text
ZS-N2N/
|-- src/
|   |-- input_utils.py          # Nguoi 1: doc anh, tao nhieu
|   |-- pair_downsampler.py     # Nguoi 1: D1, D2
|   |-- model.py                # Nguoi 2: CNN network
|   |-- losses.py               # Nguoi 2: loss_func
|   |-- train.py                # Nguoi 3: huan luyen day du
|   `-- evaluate.py             # Nguoi 3: PSNR, danh gia
|-- demos/demo_zsn2n.py         # Demo tich hop
|-- tests/test_zsn2n.py         # Kiem thu tu dong
|-- data/input/                 # Anh dau vao (khong commit dataset)
|-- data/reference/             # Anh sach tham chieu neu co
|-- outputs/images/
|-- outputs/plots/
|-- outputs/logs/
`-- reports/phan1, phan2, phan3/
```

`pair_downsampler.py` hien co ban tich hop toi thieu de loss Phan 2 chay duoc. Nguoi 1 co the kiem tra va phat trien tren nhanh rieng ma khong sua truc tiep `main`.

## Cai dat tren Windows bang VS Code

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Chay demo Phan 2

Anh nhieu tong hop, 20 vong lap mac dinh:

```powershell
py -m demos.demo_zsn2n
```

Thay doi so vong lap:

```powershell
py -m demos.demo_zsn2n --steps 50
```

Thu voi anh RGB:

```powershell
py -m demos.demo_zsn2n --image "C:\duong-dan\anh.png" --steps 20
```

Ket qua duoc ghi trong `outputs/logs`, `outputs/plots` va `outputs/images`.

## Chay kiem thu

```powershell
py -m unittest -v tests.test_zsn2n
```

Sau nhom kiem thu xac nhan: tong 22.275 tham so; CNN giu nguyen kich thuoc; hai kernel D1/D2 dung; hai thanh phan loss khop cong thuc; Total Loss bang tong hai thanh phan; gradient duoc tinh cho CNN.

## Lam viec theo nhanh

Nguoi 1:

```powershell
git checkout -b phan1-input-downsampler
```

Nguoi 3:

```powershell
git checkout -b phan3-training-evaluation
```

Moi nguoi chay kiem thu truoc khi tao Pull Request vao `main`. Khong dua `.venv`, cache Python, dataset lon hoac ket qua sinh tu dong len repository.

## Doi chieu cong thuc (2)-(6)

- Cong thuc (2) la anh xa truc tiep tu `D1(y)` sang `D2(y)`, dung de dan nhap ve ly thuyet.
- Cong thuc (3) chuyen sang residual learning: `g_theta(z) = z - f_theta(z)`.
- Cong thuc (4) la Residual Loss doi xung giua `P1` voi `D2(y)` va `P2` voi `D1(y)`.
- Cong thuc (5) dat rang buoc nhat quan cho mot phep tach.
- Cong thuc (6) ap dung rang buoc nhat quan cho ca D1 va D2.
- Total Loss bang `Residual Loss + Consistency Loss`, khong co trong so bo sung.

Ba lan goi CNN trong `loss_func` dung cung mot doi tuong model, do do dung chung mot bo tham so. Loss tu giam sat khong tu no chung minh chat luong khu nhieu; PSNR chi duoc tinh khi co anh sach tham chieu phu hop.

