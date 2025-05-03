from sample import sample_ddpm
import torch
from torch import nn
import wandb
from data_loader import get_train_loader
from model import UNetMNIST
import time
import math

# parameters
lr = 1e-4
batch_size = 64
train_epochs = 100
T = 1000
device = "cuda"
wandb_token = "7a6d6808178f08a806911ec7263c24a59f6df7da"
model_save_steps = 200
train_log_step = 10
model_save_path = "ckpts"

lr_max   = 3e-4
warmup   = 500                # steps

# Slow 
wandb.login(key=wandb_token)
wandb.init(project="ddim", config={
    "lr": lr,
    "batch_size": batch_size,
    "train_epochs": train_epochs,
    "T": T
})

start_time = time.time()
train_loader = get_train_loader()
print(f"Data load time {time.time() - start_time:.2f} seconds")


betas = torch.linspace(1e-4, 2e-2, T, device=device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar).to(device)
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar).to(device)

model = UNetMNIST()
# schedule lr 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=1e-4
)
model.to(device)

total_steps = train_epochs * len(train_loader)
schedule = lambda s: min(1, s/warmup) * 0.5 * (1 + math.cos(math.pi * s / total_steps))

total_steps = 0

from tqdm import tqdm
for epoch in range(train_epochs):
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}", unit="batch")):
        if step % 10 == 0:
            lr = schedule(total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        total_steps += 1

        x = batch["image_tensor"].to(device)
        labels = batch["label_tensor"].to(device)

        b = x.shape[0]
        t = torch.randint(0, T, (b,)).to(device)
        # noise: (b, 3, 28, 28)
        noise = torch.randn_like(x)
        noised_x = sqrt_alphas_bar[t][:,None,None,None] * x + sqrt_one_minus_alphas_bar[t][:,None,None,None] * noise
        loss = nn.functional.mse_loss(
            model(noised_x, t.float()/T), 
            noise
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % model_save_steps == 0:
            torch.save(model.state_dict(), f"{model_save_path}/model_{step}.pth")

        if step % train_log_step == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr
            })

    if epoch % 5 == 0:
        sample_ddpm(model, device=device, save_images=True)



