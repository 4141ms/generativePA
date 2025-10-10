# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：schedule.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:23 
"""
## utils/schedule.py
# utils/schedule.py

import sys
from pathlib import Path

# 将项目根目录（MySynthesisCode）加入Python路径
sys.path.append(str(Path(__file__).parent.parent))  # 若当前文件在 scripts/ 下，则 parent.parent 为根目录

import torch
import torch.nn.functional as F
from config import cfg


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(cfg.time_steps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def q_sample_latent(z0: torch.Tensor, t_idx: torch.Tensor, noise=None):
    if noise is None:
        noise = torch.randn_like(z0)
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1).to(z0.device)
    return torch.sqrt(a) * z0 + torch.sqrt(1. - a) * noise


def score_matching_loss_latent(score_net, z0, t_float, cond_z=None):
    # continuous-time denoising score matching for latent with optional conditioning
    T = cfg.time_steps
    t_idx = (t_float * (T - 1)).long()
    noise = torch.randn_like(z0)
    zt = q_sample_latent(z0, t_idx, noise)
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1).to(z0.device)
    target = - (zt - torch.sqrt(a) * z0) / (1. - a)
    if cond_z is not None:
        pred = score_net(zt, t_float, cond_z)
    else:
        pred = score_net(zt, t_float, None)
    return F.mse_loss(pred, target)
