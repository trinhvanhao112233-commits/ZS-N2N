#Hàm test: tính PSNR giữa ảnh khử nhiễu và ảnh gốc
def test(model, noisy_img, clean_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)

    return PSNR
# Hàm khử nhiễu để lấy ảnh đầu ra
def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred

denoised_img = denoise(model, noisy_img)

denoised = denoised_img.cpu().squeeze(0).permute(1,2,0)
clean = clean_img.cpu().squeeze(0).permute(1,2,0)
noisy = noisy_img.cpu().squeeze(0).permute(1,2,0)
# Gọi đánh giá sau khi train
PSNR = test(model, noisy_img, clean_img)
print(PSNR)
#Hiển thị so sánh, kèm PSNR của ảnh nhiễu
fig, ax = plt.subplots(1, 3, figsize=(15, 15))

ax[0].imshow(clean)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Ground Truth')

ax[1].imshow(noisy)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Noisy Img')
noisy_psnr = 10*np.log10(1/mse(noisy_img, clean_img).item())
ax[1].set(xlabel=str(round(noisy_psnr, 2)) + ' dB')

ax[2].imshow(denoised)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('Denoised Img')
ax[2].set(xlabel=str(round(PSNR, 2)) + ' dB')
