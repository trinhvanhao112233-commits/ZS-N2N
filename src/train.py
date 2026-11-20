#Hàm train
def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
  # Optimizer + Scheduler
max_epoch = 3000     # số epoch train
lr = 0.001           # learning rate
step_size = 1000     # cứ sau 1000 epoch thì giảm lr
gamma = 0.5          # hệ số giảm lr

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#Vòng lặp train
for epoch in tqdm(range(max_epoch)):
    train(model, optimizer, noisy_img)
    scheduler.step()
