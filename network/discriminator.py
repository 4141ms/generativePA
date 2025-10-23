# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：discriminator.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:21 
"""
## models/discriminator.py

# models/discriminator.py
import torch
import torch.nn as nn


class BoundaryDisc(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.GroupNorm(8, base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.GroupNorm(8, base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
