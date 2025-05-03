from model.dit import DiT
from sample import sample_ddpm
import torch
from torch import nn
import wandb
from data_loader import get_train_loader
from configs import mnist_config
import time
import os
from torch.nn.utils import clip_grad_norm_
import math


# parameters
init_lr = 1e-4
batch_size = 64
train_epochs = 50
T = 1000
device = "cuda"
wandb_token = "7a6d6808178f08a806911ec7263c24a59f6df7da"
model_save_steps = 200
train_log_step = 10
model_save_path = "ckpts"
sample_epochs = 5

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Slow 
wandb.login(key=wandb_token)
wandb.init(project="ddim", config={
    "init_lr": init_lr,
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

model = DiT(mnist_config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9,0.999),
)

model.to(device)

torch.compile(model)

overall_steps = train_epochs * len(train_loader)

def cosine_decay(step, total_steps, init_lr):
    # Goes from init_lr → 0 following cosine curve
    # progress = step / total_steps
    # cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return init_lr


def cosine_schedule(step):
    return cosine_decay(step, overall_steps, init_lr) if step > 500 else (step/overall_steps * init_lr)

total_steps = 0

for epoch in range(train_epochs):
    from tqdm import tqdm
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}", unit="batch")):
        total_steps += 1

        x = batch["image_tensor"].to(device)
        labels = batch["label_tensor"].to(device)

        b = x.shape[0]
        t = torch.randint(0, T, (b,)).to(device)
        # noise: (b, 3, 28, 28)
        noise = torch.randn_like(x)
        noised_x = sqrt_alphas_bar[t][:,None,None,None] * x + sqrt_one_minus_alphas_bar[t][:,None,None,None] * noise
        t_emb = t.float() / T
        # (B,)
        labels = labels
        loss = nn.functional.mse_loss(
            model(noised_x, t_emb, labels), 
            noise
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        if total_steps % model_save_steps == 0:
            torch.save(model.state_dict(), f"{model_save_path}/model_{step}.pth")
        if total_steps % train_log_step == 0:
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            wandb.log({
                "loss": loss.item(),
                "grad_norm": grad_norm
            })

    # if epoch % sample_epochs == 0:
    #     sample_ddpm(model, device=device, save_images=True)



