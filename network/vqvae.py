# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：vqvae.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:19 
"""
# Project files: complete helper modules for Conditioned Schrödinger Bridge (MRI -> PA)

# This document contains the full implementations of the helper files you requested so the modular repo is runnable.
#
# FILES INCLUDED
# - models/vqvae.py          # VQ-VAE (Encoder, Decoder, VectorQuantizer, VQVAE)
# - models/discriminator.py  # BoundaryDisc
# - utils/schedule.py        # beta schedule, q_sample_latent, conditional score-matching loss
# - utils/losses.py          # physics_consistency_loss_pixels, structure_loss
# - utils/dataset.py         # ImgDataset and optional PairedSimDataset
# - scripts/sample.py        # sampling CLI using trained models
# - scripts/train_sb.py      # fully runnable training script (overwrites prior placeholder)
# - config.py                # central configuration
#
# Each file is provided below — copy each into the matching path in your project.
#
# ---

## models/vqvae.py
# models/vqvae.py

import sys
from pathlib import Path

# 将项目根目录（MySynthesisCode）加入Python路径
sys.path.append(str(Path(__file__).parent.parent))  # 若当前文件在 scripts/ 下，则 parent.parent 为根目录

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class Encoder(nn.Module):
    def __init__(self, in_ch=cfg.channels, zc=cfg.latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, zc, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=cfg.channels, zc=cfg.latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(zc, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=cfg.codebook_size, embedding_dim=cfg.latent_dim, commitment_cost=cfg.vq_commit):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: (B,C,H,W)
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     - 2 * flat_input @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view(inputs.shape[0], inputs.shape[2], inputs.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.quant = VectorQuantizer()
        self.dec = Decoder()

    def forward(self, x):
        z = self.enc(x)
        qz, qloss = self.quant(z)
        recon = self.dec(qz)
        return recon, qz, qloss
