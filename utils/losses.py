# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：losses.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:24 
"""
## utils/losses.py
# utils/losses.py
import torch
import torch.nn.functional as F


def physics_consistency_loss_pixels(gen_pa: torch.Tensor, sim_pa_stats: dict):
    # Compare simple statistics (mean) and vesselness map if provided in sim_pa_stats
    losses = []
    if 'mean' in sim_pa_stats:
        gen_mean = gen_pa.mean(dim=[1,2,3])
        sim_mean = sim_pa_stats['mean'].to(gen_mean.device)
    losses.append(F.mse_loss(gen_mean, sim_mean))
    if 'vesselness' in sim_pa_stats:
        # sim_pa_stats['vesselness'] should be a tensor (B,1,H,W)
        v_sim = sim_pa_stats['vesselness'].to(gen_pa.device)
        # encourage similar vesselness maps (L1)
        losses.append(F.l1_loss(gen_pa, v_sim))
    if len(losses) == 0:
        return torch.tensor(0., device=gen_pa.device)
    return sum(losses) / len(losses)


def structure_loss(generated: torch.Tensor, source: torch.Tensor):
    # normalized cross-correlation (NCC) surrogate
    gen_flat = generated.view(generated.size(0), -1)
    src_flat = source.view(source.size(0), -1)
    gn = (gen_flat - gen_flat.mean(dim=1, keepdim=True))
    sn = (src_flat - src_flat.mean(dim=1, keepdim=True))
    num = (gn * sn).sum(dim=1)
    den = torch.sqrt((gn**2).sum(dim=1) * (sn**2).sum(dim=1) + 1e-8)
    ncc = num / (den + 1e-8)
    return torch.mean(1. - ncc)