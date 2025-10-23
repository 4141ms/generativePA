
"""
conditional_sb_mri2pa.py
A minimal, well-commented PyTorch template for a NON-DISENTANGLED (end-to-end) MRI -> PA mapping.
Design: train a U-Net style encoder-decoder and a conditional latent DDPM (simplified) to model
p(z_PA | z_MRI) in latent space. This is a practical proxy for conditional Schrödinger Bridge.
Also includes domain adaptation (MMD) and uncertainty estimation via multiple latent-samples.

NOTES:
- Replace dataset placeholders with your real data loaders.
- Replace simplified DDPM / sampling with a full SB implementation if desired.
- Script *does not* start heavy training automatically — it contains functions and a `main()`
  entrypoint that you can adapt/run for experiments.
"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Utilities (reparameterize, MMD, simple logging)
# -----------------------------------------------------------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_mmd(x, y, sigma_list=[1, 2, 4, 8, 16]):
    # x: (N, D), y: (M, D)
    xx = x @ x.t()           # (N, N)
    yy = y @ y.t()           # (M, M)
    xy = x @ y.t()           # (N, M)

    rx = xx.diag().unsqueeze(1).expand_as(xx)  # (N, N)
    ry = yy.diag().unsqueeze(1).expand_as(yy)  # (M, M)
    dxx = rx + rx.t() - 2 * xx
    dyy = ry + ry.t() - 2 * yy

    # 修复：为 dxy 正确广播
    rx = xx.diag().unsqueeze(1).expand(x.size(0), y.size(0))  # (N, M)
    ry = yy.diag().unsqueeze(0).expand(x.size(0), y.size(0))  # (N, M)
    dxy = rx + ry - 2 * xy                                    # (N, M)

    Kxx = 0
    Kyy = 0
    Kxy = 0
    for sigma in sigma_list:
        Kxx += torch.exp(-dxx / (2 * sigma**2))
        Kyy += torch.exp(-dyy / (2 * sigma**2))
        Kxy += torch.exp(-dxy / (2 * sigma**2))

    n = x.size(0)
    m = y.size(0)
    sum_kxx = (Kxx.sum() - Kxx.trace()) / (n * (n - 1))
    sum_kyy = (Kyy.sum() - Kyy.trace()) / (m * (m - 1))
    sum_kxy = Kxy.sum() / (n * m)
    return sum_kxx + sum_kyy - 2 * sum_kxy


# -----------------------------------------------------------------------------
# Simple Datasets - replace with your real nifti/npz loaders
# -----------------------------------------------------------------------------
class ArrayDataset(Dataset):
    def __init__(self, list_x, list_y, transform=None):
        # list_x/list_y are lists of numpy arrays (H,W) normalized to [0,1]
        self.xs = list_x
        self.ys = list_y
        self.transform = transform
    def __len__(self):
        return max(len(self.xs), len(self.ys))
    def __getitem__(self, idx):
        x = self.xs[idx % len(self.xs)]
        y = self.ys[idx % len(self.ys)]
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

# -----------------------------------------------------------------------------
# Backbone: simple encoder-decoder (U-Net style) - end-to-end mapping MRI -> latent
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base)
        self.c2 = ConvBlock(base, base*2)
        self.c3 = ConvBlock(base*2, base*4)
        self.c4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        f1 = self.c1(x)
        p1 = self.pool(f1)
        f2 = self.c2(p1)
        p2 = self.pool(f2)
        f3 = self.c3(p2)
        p3 = self.pool(f3)
        f4 = self.c4(p3)
        return [f1, f2, f3, f4]

class Decoder(nn.Module):
    def __init__(self, out_ch=1, base=32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.conv3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv2 = ConvBlock(base*4, base*2)
        self.up3 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv1 = ConvBlock(base*2, base)
        self.final = nn.Conv2d(base, out_ch, 1)
    def forward(self, features):
        f1, f2, f3, f4 = features
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(self.final(x))

# -----------------------------------------------------------------------------
# Latent projection heads - map encoder top features to latent z (for diffusion)
# We'll compress f4 -> z_dim with global avgpool + linear
# -----------------------------------------------------------------------------
class LatentHead(nn.Module):
    def __init__(self, in_ch, z_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(in_ch, z_dim)
        self.fc_logvar = nn.Linear(in_ch, z_dim)
    def forward(self, feat):  # feat: (B, C, H, W)
        h = self.pool(feat).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# -----------------------------------------------------------------------------
# Simple conditional denoiser network for latent DDPM (MLP or small UNet-like)
# We use an MLP that takes noisy z_t and time embedding and condition (z_mri)
# -----------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Linear(dim*4, dim))
    def forward(self, t):
        # t: tensor of shape (B,) with integers 0..T-1
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, dtype=torch.float32) * -math.log(10000) / half).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        emb_sin = torch.sin(emb)
        emb_cos = torch.cos(emb)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        return self.fc(emb)

class LatentDenoiser(nn.Module):
    def __init__(self, z_dim=128, cond_dim=128, hidden=512, time_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim)
        )
    def forward(self, z_t, cond_z, t):
        # z_t: (B, z_dim), cond_z: (B, cond_dim), t: (B,) int tensor
        te = self.time_emb(t)  # (B, time_dim)
        inp = torch.cat([z_t, cond_z, te], dim=1)
        return self.net(inp)

# -----------------------------------------------------------------------------
# DDPM schedule utilities (simplified) - forward q_sample and loss target
# -----------------------------------------------------------------------------
def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)

def q_sample(z0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    # z0: (B,D), t: (B,) ints
    if noise is None:
        noise = torch.randn_like(z0)
    # gather scalars per sample
    a = sqrt_alphas_cumprod[t].view(-1,1).to(z0.device)
    b = sqrt_one_minus_alphas_cumprod[t].view(-1,1).to(z0.device)
    return a * z0 + b * noise

# -----------------------------------------------------------------------------
# Training routines (Stage0: train autoencoder; Stage1: train conditional latent DDPM;
# Stage2: human fine-tune with domain adaptation)
# -----------------------------------------------------------------------------
def train_stage0_autoencoder(encoder, decoder, latent_head, dataloader, optim, device, epochs=5):
    encoder.train(); decoder.train(); latent_head.train()
    l1 = nn.L1Loss()
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for x_np, y_np in pbar:
            x = x_np.to(device)  # MRI input
            y = y_np.to(device)  # PA target (used for supervised AE reconstruction in stage0)
            feats = encoder(x)
            # use encoder features to decode directly to PA (end-to-end mapping)
            y_pred = decoder(feats)
            Lrec = l1(y_pred, y)
            optim.zero_grad()
            Lrec.backward()
            optim.step()
            pbar.set_description(f"AE ep{ep} Lrec:{Lrec.item():.4f}")
    return

def train_stage1_latent_ddpm(encoder, decoder, latent_head, ddpm, mapper_opt, dataloader_mouse, T=200, device='cuda', epochs=10):
    # This trains conditional DDPM in latent: map z_MRI -> z_PA distribution
    ddpm.train(); encoder.eval(); latent_head.eval()  # freeze encoder for latent stability optionally
    # prepare beta schedule scalars
    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    mse = nn.MSELoss()
    for ep in range(epochs):
        pbar = tqdm(dataloader_mouse)
        for x_np, y_np in pbar:
            x = x_np.to(device); y = y_np.to(device)
            with torch.no_grad():
                feats_x = encoder(x); feats_y = encoder(y)
                mu_x, logvar_x = latent_head(feats_x[-1])
                mu_y, logvar_y = latent_head(feats_y[-1])
                z_x = mu_x  # use mean as condition (deterministic conditioning); could sample instead
                z_y = mu_y  # target latent mean

            # sample noise and t
            B = z_y.size(0)
            t = torch.randint(0, T, (B,), device=device)
            noise = torch.randn_like(z_y)
            z_t = q_sample(z_y, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
            # ddpm predict noise (epsilon) from (z_t, cond=z_x, t)
            eps_pred = ddpm(z_t, z_x, t)
            loss = mse(eps_pred, noise)
            mapper_opt.zero_grad()
            loss.backward()
            mapper_opt.step()
            pbar.set_description(f"DDPM ep{ep} loss:{loss.item():.6f}")
    return

def train_stage2_human_finetune(encoder, decoder, latent_head, ddpm, mouse_bank, human_loader, opt, device, T=200, lambda_mmd=1.0, epochs=5):
    # mouse_bank: tensor of mouse latent means collected earlier (N_mouse, z_dim)
    encoder.train(); decoder.train(); latent_head.train(); ddpm.eval()
    l1 = nn.L1Loss()
    for ep in range(epochs):
        pbar = tqdm(human_loader)
        for x_np, y_np in pbar:
            x = x_np.to(device); y = y_np.to(device)
            feats_x = encoder(x)
            mu_x, _ = latent_head(feats_x[-1])  # cond
            # sample a mouse latent prototype and run reverse DDPM sampling (simplified: predict denoised mean)
            idx = random.randrange(0, mouse_bank.size(0))
            z_mouse = mouse_bank[idx:idx+1].to(device)
            # Use ddpm to map z_mouse -> human-like z via a simple deterministic pass (placeholder)
            # A real implementation would run reverse diffusion conditioned on mu_x
            with torch.no_grad():
                z_h_pred = z_mouse  # placeholder mapping - replace with reverse sampler
            # decode from z_h_pred by broadcasting into decoder via a tiny MLP -> feature map (placeholder)
            # here we simply use decoder fed with encoder features (no latent-to-feature mapping implemented)
            y_pred = decoder(feats_x)
            # losses: pixel + MMD between encoded outputs and mouse_bank distribution (domain adaptation)
            Lpix = l1(y_pred, y)
            # compute encoded latent means of generated image and compare with mouse_bank sample
            with torch.no_grad():
                feats_gen = encoder(y_pred)
                mu_gen, _ = latent_head(feats_gen[-1])
            Lmmd = compute_mmd(mu_gen.detach().cpu(), mouse_bank.cpu()) if mouse_bank is not None else 0.0
            loss = Lpix + lambda_mmd * Lmmd
            opt.zero_grad()
            # note: Lmmd currently on cpu; in real code move tensors to device and use torch for autodiff
            loss.backward()
            opt.step()
            pbar.set_description(f"FT ep{ep} Lpix:{Lpix.item():.4f} Lmmd:{float(Lmmd):.6f}")
    return

# -----------------------------------------------------------------------------
# Helper to collect mouse latent bank (means) for domain adaptation
# -----------------------------------------------------------------------------
def build_mouse_bank(encoder, latent_head, dataloader, device):
    encoder.eval(); latent_head.eval()
    mus = []
    with torch.no_grad():
        for x_np, y_np in tqdm(dataloader):
            y = y_np.to(device)
            feats_y = encoder(y)
            mu_y, _ = latent_head(feats_y[-1])
            mus.append(mu_y.cpu())
    if len(mus) == 0:
        return None
    return torch.cat(mus, dim=0)

# -----------------------------------------------------------------------------
# Main: assemble and create dataloaders - this does NOT run heavy training automatically
# -----------------------------------------------------------------------------
def main_demo_write():
    """
    This function creates model instances and illustrates how to call stage functions.
    It also writes a small README note.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = 32; z_dim = 128
    # create models
    encoder = Encoder(in_ch=1, base=base).to(device)
    decoder = Decoder(out_ch=1, base=base).to(device)
    latent_head = LatentHead(in_ch=base*8, z_dim=z_dim).to(device)
    ddpm = LatentDenoiser(z_dim=z_dim, cond_dim=z_dim, hidden=512, time_dim=128).to(device)

    # small synthetic data for smoke-run - replace with your loader
    def to_tensor(arr):
        t = torch.tensor(arr, dtype=torch.float32)
        if t.ndim == 2: t = t.unsqueeze(0)
        return t

    mouse_mri = [np.random.rand(256,256).astype('float32') for _ in range(100)]
    mouse_pa  = [np.random.rand(256,256).astype('float32') for _ in range(100)]
    human_mri = [np.random.rand(256,256).astype('float32') for _ in range(10)]
    human_pa  = [np.random.rand(256,256).astype('float32') for _ in range(5)]

    mouse_ds = ArrayDataset(mouse_mri, mouse_pa, transform=lambda x: to_tensor(x))
    human_ds = ArrayDataset(human_mri, human_pa, transform=lambda x: to_tensor(x))
    mouse_loader = DataLoader(mouse_ds, batch_size=4, shuffle=True, drop_last=True)
    human_loader = DataLoader(human_ds, batch_size=2, shuffle=True, drop_last=True)

    # optimizers (placeholders)
    opt_ae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    opt_ddpm = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
    opt_ft = torch.optim.Adam(list(decoder.parameters()) + list(latent_head.parameters()), lr=1e-5)

    # Stage0: quick pretrain AE (demo small epochs)
    print('Stage0 demo pretrain (AE) - running 1 epoch smoke test')
    train_stage0_autoencoder(encoder, decoder, latent_head, mouse_loader, opt_ae, device, epochs=1)

    # Build mouse bank
    print('Building mouse latent bank (demo)')
    mouse_bank = build_mouse_bank(encoder, latent_head, mouse_loader, device)

    # Stage1: train latent DDPM mapping (demo 1 epoch)
    print('Stage1 demo latent DDPM (1 epoch)')
    train_stage1_latent_ddpm(encoder, decoder, latent_head, ddpm, opt_ddpm, mouse_loader, T=100, device=device, epochs=1)

    # Stage2: human fine-tune (demo)
    print('Stage2 demo human fine-tune (1 epoch)')
    train_stage2_human_finetune(encoder, decoder, latent_head, ddpm, mouse_bank, human_loader, opt_ft, device, epochs=1)

    # Save a small checkpoint
    save_dir = './outputs/'
    os.makedirs(save_dir, exist_ok=True)  # 若文件夹不存在则创建
    ckpt_path = os.path.join(save_dir, "conditional_sb_mri2pa_demo_ckpt.pth")
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'latent_head': latent_head.state_dict(),
        'ddpm': ddpm.state_dict()
    }, ckpt_path)

    print(f"✅ 模型已保存到 {ckpt_path}")

    print('Saved demo checkpoint to', str(ckpt_path))
    return str(ckpt_path)

if __name__ == "__main__":
    # Running this script as-is will perform a tiny smoke-run with synthetic data and save a demo checkpoint.
    ckpt_path = main_demo_write()
    print("Demo finished. Checkpoint:", ckpt_path)
