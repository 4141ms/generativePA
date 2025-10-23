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

    def __init__(self, nVSSMs=2, embed_dim=256, patch_size=8, d_state=16, expand=2.,
                 compress_ratio=8, squeeze_factor=8, mamba_from_trion=0):
        super().__init__()

        net = [PatchEmbedNN(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)]
        for _ in range(nVSSMs):
            net.append(
                VSSM(hidden_dim=embed_dim, d_state=d_state, expand=expand, mamba_from_trion=mamba_from_trion))
        net.append(LoE(num_feat=embed_dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        feat = self.net(x)
        return feat


if __name__ == '__main__':
    import torch

    net = VSSMEncoder(embed_dim=64, patch_size=8, d_state=16, expand=2.,
                      compress_ratio=8, squeeze_factor=8, mamba_from_trion=0).cuda()

    # embed_dim 需要小于 H, W
    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    c = torch.randn((1, 3, 128, 128)).cuda()
    out = net.forward(c)
    print(out.shape)
