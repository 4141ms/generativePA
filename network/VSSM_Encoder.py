import torch.nn as nn

from model.Patch_EmbedNN import PatchEmbedNN
from model.StyleEmbedder import GlobalStyleEncoder
from model.VSSM import VSSM
from model.SAVSSG import SAVSSG
from model.Decoder import Decoder_NN, Decoder_NN_x4
from model.LoE import LoE


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


class VSSMEncoder(nn.Module):
    """
    如果有安装mamba_ssm设置:mamba_from_trion=1
    embed_dim需要小于 H, W
    """

    def __init__(self, in_c=15, nVSSMs=2, embed_dim=64, patch_size=8, d_state=8, expand=2.,
                 compress_ratio=8, squeeze_factor=8, mamba_from_trion=0):
        super().__init__()

        net = [PatchEmbedNN(patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim)]
        for _ in range(nVSSMs):
            net.append(
                VSSM(hidden_dim=embed_dim, d_state=d_state, expand=expand, mamba_from_trion=mamba_from_trion))
        net.append(LoE(num_feat=embed_dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        feat = self.net(x)
        return feat

class DisentangledEncoder(nn.Module):
    """
    需要用解码特征
    """
    def __init__(self):
        super().__init__()
        # 共享内容编码器（提取解剖结构）
        self.content_enc = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # ResBlock(64)  #  这里要再写
        )

        # 模态特有编码器（提取风格特征）
        self.style_enc_mri = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256)
        )
        self.style_enc_pa = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256)
        )

    def forward(self, x, modal_type):
        content = self.content_enc(x)
        if modal_type == 'mri':
            style = self.style_enc_mri(content)
        else:
            style = self.style_enc_pa(content)
        return content, style  # 返回解耦特征

if __name__ == '__main__':
    import torch

    net = VSSMEncoder(in_c=5).cuda()

    # embed_dim 需要小于 H, W
    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    c = torch.randn((1, 5, 256, 256)).cuda()
    out = net.forward(c) # [B, embed_dim, H/patch_size, W/patch_size]
    print(out.shape)
