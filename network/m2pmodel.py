# main.py 或你的脚本最上方
import warnings

from utils.losses import UncertaintyLoss

# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import pytorch_lightning as pl
from torchvision.utils import make_grid
from model.BrownianBridge.BrownianBridge import BrownianBridgeModel
from model.BrownianBridge.Restormer import Restormer
from model.FiLMFusion import FiLMFusion, FiLMFusionWithAttention
from model.UNSB.sb_model import SBModel
from pytorch_lightning.utilities import rank_zero_only

import pytorch_lightning as pl
import torch
from network.VSSM_Encoder import VSSMEncoder
from network.Decoder import DecoderMultiHead
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

class M2PModel(nn.Module):
    """
    Lightning wrapper for M2PModel using UNSB (SBModel) and optional BBDM.
    Supports manual optimization for multiple optimizers.
    """
    def __init__(self, options):
        super().__init__()
        self.options = options
        # self.use_modality = options.data.use_modality

        self.content_encoder1 = VSSMEncoder(in_c=options.data.use_slice)
        self.content_encoder2 = VSSMEncoder(in_c=options.data.use_slice)
        self.content_encoder3 = VSSMEncoder(in_c=options.data.use_slice)

        self.style_encoder1 = VSSMEncoder(in_c=options.data.use_slice)
        self.style_encoder2 = VSSMEncoder(in_c=options.data.use_slice)
        self.style_encoder3 = VSSMEncoder(in_c=options.data.use_slice)

        self.target_feature = VSSMEncoder(in_c=options.data.use_slice)

        enc_dim = 64
        self.content_fusion = FiLMFusion(feat_dim=enc_dim, use_cross_attn=True, num_heads=4)
        self.style_fusion = FiLMFusion(feat_dim=enc_dim, use_cross_attn=True, num_heads=4)

        self.bridge = SBModel(options.UNSB)

        self.decoder = DecoderMultiHead(feature_channel=enc_dim, out_size=256, slice_num=options.data.use_slice)




    def initialize_bridge(self, batch):
        # 使用一个 batch 初始化 netF
        target_f = self.target_feature(batch['PA'])

        c_f1 = self.content_encoder1(batch['T1'])
        c_f2 = self.content_encoder2(batch['T2'])
        c_f3 = self.content_encoder3(batch['MRA'])

        s_f1 = self.style_encoder1(batch['T1'])
        s_f2 = self.style_encoder2(batch['T2'])
        s_f3 = self.style_encoder3(batch['MRA'])

        c_fusion = self.content_fusion([c_f1, c_f2, c_f3])  # torch.Size([2, 64, 32, 32])
        s_fusion = self.style_fusion([s_f1, s_f2, s_f3])  # torch.Size([2, 64, 32, 32])

        self.bridge.data_dependent_initialize(c_fusion, target_f, s_fusion)


    def forward(self, batch):
        target_f = self.target_feature(batch['PA'])

        c_f1 = self.content_encoder1(batch['T1'])
        c_f2 = self.content_encoder2(batch['T2'])
        c_f3 = self.content_encoder3(batch['MRA'])

        s_f1 = self.style_encoder1(batch['T1'])
        s_f2 = self.style_encoder2(batch['T2'])
        s_f3 = self.style_encoder3(batch['MRA'])

        c_fusion = self.content_fusion([c_f1, c_f2, c_f3]) # torch.Size([2, 64, 32, 32])
        s_fusion = self.style_fusion([s_f1, s_f2, s_f3]) # torch.Size([2, 64, 32, 32])

        # 设置输入给 bridge
        self.bridge.set_input(c_fusion, target_f, s_fusion)
        # 前向计算
        self.bridge.forward()

        out_latent_f = self.bridge.fake_B
        p0, c, rho, sigma2 = self.decoder(out_latent_f)


        # 返回生成图像
        return p0, c, rho, sigma2
