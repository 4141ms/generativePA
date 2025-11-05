# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：losses.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:24 
"""
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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

def gumbel_softmax_sample(logits, temperature, gumbel, dim):
    '''mip loss'''
    y = logits + gumbel
    return F.softmax(y / temperature, dim)


def cal_snr(noise_img, clean_img):
    noise_img, clean_img = noise_img.detach().cpu().numpy(), clean_img.detach().cpu().numpy()
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


class MIPloss(nn.Module):
    """ mmdm"""
    def __init__(self, options):
        super().__init__()
        self.temp = options.mip_loss.temp
        self.num_slice = options.data.use_slice
        # self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()

    def reset_gumbel(self, img_fake):
        U = torch.rand_like(img_fake)
        # U = torch.rand(img_fake.size()).cuda()
        self.gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)  # sample_gumbel

    def forward(self, img_fake, batch):
        self.reset_gumbel(img_fake)
        target = batch['PA']
        pred_mips_c1 = torch.zeros_like(img_fake)
        target_mips_c1 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, :idx + 1], self.temp, self.gumbel[:, :idx + 1], dim=1)
            target_mips_c1[:, idx] = torch.max(target[:, :idx + 1], dim=1)[0]
            pred_mips_c1[:, idx] = torch.sum(pred_mip * img_fake[:, :idx + 1], dim=1)

        pred_mips_c2 = torch.zeros_like(img_fake)
        target_mips_c2 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, self.num_slice - idx - 1:], self.temp,
                                             self.gumbel[:, self.num_slice - idx - 1:], dim=1)
            target_mips_c2[:, idx] = torch.max(target[:, self.num_slice - idx - 1:], dim=1)[0]
            pred_mips_c2[:, idx] = torch.sum(pred_mip * img_fake[:, self.num_slice - idx - 1:], dim=1)

        loss_ = self.L2(img_fake, target)
        loss_mip_c1 = self.L2(pred_mips_c1, target_mips_c1)
        loss_mip_c2 = self.L2(pred_mips_c2, target_mips_c2)
        loss = loss_ + loss_mip_c1 + loss_mip_c2

        return loss


class AutomaticWeightedLoss(nn.Module):
    """ mmdm"""
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, losses, sigma_t):
        loss_sum = 0

        for i, loss in enumerate(losses):
            if i != 0:
                adjust_para = self.params[i] ** 2 + sigma_t
            else:
                adjust_para = self.params[i] ** 2
            loss_sum += 0.5 / adjust_para * loss + torch.log(1 + adjust_para)

        return loss_sum


import torch
import torch.nn.functional as F

class UncertaintyLoss(torch.nn.Module):
    """
    适用于 DecoderMultiHead 输出的损失函数
    p0, c, rho: [B,1,H,W]
    sigma2: [B,3,H,W]
    """

    def __init__(self, lambda_mean=1.0, lambda_nll=1.0, eps=1e-6):
        super().__init__()
        self.lambda_mean = lambda_mean  # 均值损失权重
        self.lambda_nll = lambda_nll    # NLL损失权重
        self.eps = eps

    def nll_loss(self, pred, target, sigma2):
        """
        高斯负对数似然
        sigma2: 方差, 需 >=0
        """
        loss = 0.5 * ((target - pred) ** 2 / (sigma2 + self.eps) + torch.log(sigma2 + self.eps))
        return loss.mean()

    def forward(self, p0, c, rho, sigma2, target_p0, target_c, target_rho):
        # --- 均值损失 ---
        loss_mean = F.mse_loss(p0, target_p0) + F.mse_loss(c, target_c) + F.mse_loss(rho, target_rho)

        # --- 不确定性损失 ---
        loss_nll = self.nll_loss(p0, target_p0, sigma2[:,0:1]) + \
                   self.nll_loss(c, target_c, sigma2[:,1:2]) + \
                   self.nll_loss(rho, target_rho, sigma2[:,2:3])

        # --- 总损失 ---
        loss = self.lambda_mean * loss_mean + self.lambda_nll * loss_nll
        return loss, loss_mean, loss_nll


