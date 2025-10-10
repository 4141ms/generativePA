# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：score_net.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:54 
"""
## File: models/score_net.py (ADDED)

# models/score_net.py
"""
Conditional Latent-space Score Network (LatentScoreNet)
- Supports classifier-free guidance via conditioning dropout
- Exposes utilities for guidance sampling
- Designed to operate on quantized/continuous latents (B, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import cfg


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(0, half, device=device) / float(half))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1), mode='constant', value=0)
        return emb


class LatentConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_emb_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.cond_mlp = None
        if cond_emb_dim is not None:
            self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, out_ch))

    def forward(self, x, t_emb=None, cond_emb=None):
        h = self.conv(x)
        if self.time_mlp is not None and t_emb is not None:
            te = self.time_mlp(t_emb).view(x.size(0), -1, 1, 1)
            h = h + te
        if self.cond_mlp is not None and cond_emb is not None:
            ce = self.cond_mlp(cond_emb).view(x.size(0), -1, 1, 1)
            h = h + ce
        return h


class LatentScoreNet(nn.Module):
    def __init__(self, in_ch=cfg.latent_dim, base_ch=128, time_emb_dim=256, cond_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        # project conditional latent to compact vector
        self.cond_proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, cond_dim), nn.SiLU())

        self.enc1 = LatentConvBlock(in_ch, base_ch // 2, time_emb_dim, cond_dim)
        self.enc2 = LatentConvBlock(base_ch // 2, base_ch, time_emb_dim, cond_dim)
        self.enc3 = LatentConvBlock(base_ch, base_ch, time_emb_dim, cond_dim)
        self.mid = LatentConvBlock(base_ch, base_ch, time_emb_dim, cond_dim)
        self.dec3 = LatentConvBlock(base_ch * 2, base_ch, time_emb_dim, cond_dim)
        self.dec2 = LatentConvBlock(base_ch * 2, base_ch // 2, time_emb_dim, cond_dim)
        self.dec1 = LatentConvBlock(base_ch, in_ch, time_emb_dim, cond_dim)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, z, t, cond_z: Optional[torch.Tensor] = None):
        """
        z: latent to denoise (B, C, H, W)
        t: time floats (B,)
        cond_z: conditioning latent (B, C_cond, H_cond, W_cond) or None
        returns: predicted score (same shape as z)
        """
        t_emb = self.time_mlp(t)
        cond_emb = None
        if cond_z is not None:
            cond_emb = self.cond_proj(cond_z)
        e1 = self.enc1(z, t_emb, cond_emb)
        e2 = self.enc2(self.pool(e1), t_emb, cond_emb)
        e3 = self.enc3(self.pool(e2), t_emb, cond_emb)
        m = self.mid(self.pool(e3), t_emb, cond_emb)
        d3 = self.up(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, t_emb, cond_emb)
        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb, cond_emb)
        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        out = self.dec1(d1, t_emb, cond_emb)
        return out


# Classifier-free guidance helper: randomly drop cond during training
def conditioning_dropout(cond_z: Optional[torch.Tensor], drop_prob=0.1):
    if cond_z is None:
        return None
    if not torch.is_tensor(cond_z):
        return cond_z
    mask = (torch.rand(cond_z.size(0), device=cond_z.device) > drop_prob).float().view(-1, 1, 1, 1)
    return cond_z * mask


# Sampling helper with guidance
@torch.no_grad()
def sample_conditional(score_net: LatentScoreNet, vqvae, cond_z: torch.Tensor, steps=200, guidance_scale=1.0,
                       batch=None, device=None):
    """
    Performs reverse SDE sampling in latent space with classifier-free guidance.
    - cond_z: (B, C, H, W) conditioning MRI latent
    - guidance_scale: >1 amplifies conditional score
    Returns decoded images and final latent.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    B = cond_z.size(0) if batch is None else batch
    z = torch.randn(B, cfg.latent_dim, cfg.z_h, cfg.z_h, device=device)
    for i in reversed(range(steps)):
        t = torch.full((B,), float(i) / float(steps - 1), device=device)
        # compute conditional and unconditional scores
        s_cond = score_net(z, t, cond_z)
        s_uncond = score_net(z, t, None)
        s = s_uncond + guidance_scale * (s_cond - s_uncond)
        beta = torch.linspace(1e-4, 0.02, cfg.time_steps, device=device)[i]
        z = z + 0.5 * beta * s
        if i > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)
    recon = vqvae.dec(z)
    return recon, z
