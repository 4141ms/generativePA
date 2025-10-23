import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProjectionHead(nn.Module):
    def __init__(self, in_channels, z_dim, hidden_dim=None, norm_type="layer", dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim

        # 1. 全局特征聚合
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2. 投影网络（支持单层/多层）
        hidden_dim = hidden_dim or in_channels  # 默认隐藏层维度等于输入通道数
        layers = []

        # 可选隐藏层
        if hidden_dim != in_channels:
            layers.extend([
                nn.Linear(in_channels, hidden_dim),
                self._get_norm_layer(norm_type, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])

        # 最终投影层
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.proj = nn.Sequential(*layers)

        # 3. 输出初始化
        self._init_weights()

    def _get_norm_layer(self, norm_type, dim):
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        elif norm_type == "batch":
            return nn.BatchNorm1d(dim)
        else:
            return nn.Identity()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        x = self.gap(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        z = self.proj(x)  # [B, z_dim]

        # 可选: L2归一化（适用于对比学习等场景）
        # return F.normalize(z, dim=-1) if self.normalize else z

        return z