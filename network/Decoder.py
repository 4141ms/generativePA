from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderMultiHead(nn.Module):
    """
    Multi-head decoder: 输出 p0, c, rho 和 sigma^2
    输入: latent [B, feature_channel, H, W]，H=W=32
    输出:
        p0, c, rho: [B, slice_num, 256, 256]
        sigma2: [B, 3*slice_num, 256, 256]
    """

    def __init__(self, feature_channel=64, out_size=256, slice_num=8):
        super().__init__()
        self.slice_num = slice_num

        # shared feature extraction
        self.shared = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel, feature_channel // 2, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 2, feature_channel // 4, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 4, feature_channel // 4, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 4, feature_channel // 8, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_channel // 8, feature_channel // 8, 3),
            nn.ReLU(),
        )

        # 分支输出
        self.p0_head = nn.Conv2d(feature_channel // 8, slice_num, 3, padding=1)
        self.c_head = nn.Conv2d(feature_channel // 8, slice_num, 3, padding=1)
        self.rho_head = nn.Conv2d(feature_channel // 8, slice_num, 3, padding=1)
        self.sigma_head = nn.Conv2d(feature_channel // 8, 3 * slice_num, 3, padding=1)  # 每个物理量 slice_num 通道不确定性

        self.out_size = out_size

    def forward(self, x):
        h = self.shared(x)  # [B, feature_channel//8, H_up, W_up]

        # 分支输出
        p0 = self.p0_head(h)       # [B, slice_num, H, W]
        c = self.c_head(h)         # [B, slice_num, H, W]
        rho = self.rho_head(h)     # [B, slice_num, H, W]
        sigma2 = self.sigma_head(h) # [B, 3*slice_num, H, W]

        # 上采样到目标尺寸
        p0 = F.interpolate(p0, size=self.out_size, mode='bilinear', align_corners=False)
        c = F.interpolate(c, size=self.out_size, mode='bilinear', align_corners=False)
        rho = F.interpolate(rho, size=self.out_size, mode='bilinear', align_corners=False)
        sigma2 = F.interpolate(sigma2, size=self.out_size, mode='bilinear', align_corners=False)

        # 非负保证
        p0 = F.softplus(p0)
        c = F.softplus(c)
        rho = F.softplus(rho)
        sigma2 = F.softplus(sigma2)

        return p0, c, rho, sigma2


if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DecoderMultiHead(64, 256, 5).to(device)

    latent = torch.randn((2, 64, 32, 32)).to(device)
    p0, c, rho, sigma2 = net.forward(latent)

    print(p0.shape, c.shape, rho.shape, sigma2.shape)
    print(p0.max(), p0.min())
    print(c.max(), c.min())
    print(rho.max(), rho.min())
    print(sigma2.max(), sigma2.min())