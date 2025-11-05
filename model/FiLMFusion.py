import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class FiLMFusion(nn.Module):
    """
    后续可以再改进
    Multi-modal fusion block:
      - FiLM cross-modulation (each modality is modulated by aggregated other-modals)
      - Adaptive modality weighting (scalar weights via softmax)
      - Concatenate -> 1x1 conv to reduce channels
      - Channel attention (SE-style) + Spatial attention (CBAM-style)
      - Optional Cross-Attention (MultiheadAttention) over flattened spatial tokens

    Inputs:
      feats: List[Tensor], each Tensor shape [B, C, H, W] and same C/H/W for all modalities.
    Args:
      feat_dim: int, per-modality channel C
      reduction: int, channel reduction for SE
      film_hidden: int, hidden dim multiplier when generating gamma/beta (default uses 1x1 conv)
      use_cross_attn: bool, whether to apply transformer cross-attention
      num_heads: int, for MultiheadAttention
    Output:
      fused: Tensor [B, feat_dim, H, W]
    """
    def __init__(self,
                 feat_dim: int,
                 reduction: int = 8,
                 film_hidden: Optional[int] = None,
                 use_cross_attn: bool = True,
                 num_heads: int = 4,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.C = feat_dim
        self.reduction = reduction
        self.use_cross_attn = use_cross_attn

        if film_hidden is None:
            film_hidden = feat_dim  # simple default

        # FiLM generator: map conditioning feature -> (gamma, beta) for target channels
        # We will use adaptive pool + 1x1 conv; reuse same module for all cond->target combos
        self.film_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, film_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(film_hidden, feat_dim * 2, kernel_size=1)  # produces gamma and beta
        )

        # Modality weight generator: per-modality scalar
        # takes pooled feature and outputs a scalar score
        self.mod_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=1)  # scalar per modality
        )

        # After concatenation of modulated modalities -> reduce back to feat_dim
        self.fuse_conv = nn.Conv2d(feat_dim * 1, feat_dim, kernel_size=1)  # placeholder; will use dynamic conv later

        # Channel attention (SE-style)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // reduction, feat_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention (CBAM-style)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        # Cross-attention (optional): use MultiheadAttention on flattened tokens
        if use_cross_attn:
            # MultiheadAttention expects embed_dim, and seq_len first or batch_first True (PyTorch >= 1.11)
            # We'll use batch_first=True for convenience.
            self.num_heads = num_heads
            self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads,
                                                    dropout=attn_dropout, batch_first=True)
            # small FFN after attn
            self.attn_ffn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim)
            )
            self.attn_norm1 = nn.LayerNorm(feat_dim)
            self.attn_norm2 = nn.LayerNorm(feat_dim)

        # Initialize fuse_conv dynamically in forward depending on number of modalities
        self._fuse_conv_initialized_for_n = None

    def _ensure_fuse_conv(self, n_mods: int, device):
        # If not initialized for this n, recreate fuse_conv to accept correct channels
        if self._fuse_conv_initialized_for_n == n_mods:
            return
        in_ch = self.C * n_mods
        self.fuse_conv = nn.Conv2d(in_ch, self.C, kernel_size=1).to(device)
        self._fuse_conv_initialized_for_n = n_mods

    def film_modulate(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Modulate x by cond using FiLM parameters generated from cond.
        x, cond: [B, C, H, W]
        returns: gamma * x + beta, where gamma/beta shape [B, C, 1, 1]
        """
        gb = self.film_gen(cond)  # [B, 2C, 1, 1]
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma * x + beta

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: list of tensors [B, C, H, W], same C/H/W all
        """
        assert isinstance(feats, list) and len(feats) >= 1
        B, C, H, W = feats[0].shape
        device = feats[0].device
        for f in feats:
            assert f.shape[1] == C and f.shape[2] == H and f.shape[3] == W

        N = len(feats)
        self._ensure_fuse_conv(N, device)

        # === 1) FiLM cross-modulation ===
        # For each modality i, condition on aggregated other-mods:
        modulated = []
        # Precompute sum of pooled conds to make conditioning stable:
        # We'll use simple strategy: cond_i = mean of other modality feature maps (elementwise mean)
        for i in range(N):
            # aggregate others by elementwise mean (could be sum/concat alternative)
            others = [feats[j] for j in range(N) if j != i]
            if len(others) == 0:
                cond = feats[i]  # fallback
            else:
                # simple elementwise mean
                cond = torch.stack(others, dim=0).mean(dim=0)
            mod_i = self.film_modulate(feats[i], cond)  # [B,C,H,W]
            modulated.append(mod_i)

        # === 2) Modality adaptive weighting (scalar weights) ===
        # compute unnormalized scores per modality
        scores = []
        for m in modulated:
            s = self.mod_weight_gen(m)  # [B,1,1,1]
            s = s.view(B, 1)           # [B,1]
            scores.append(s)
        scores = torch.cat(scores, dim=1)   # [B, N]
        weights = F.softmax(scores, dim=1)  # [B, N]
        # apply weights and sum? we'll apply weights then concat to preserve per-modality channels
        weighted_mods = []
        for idx, m in enumerate(modulated):
            w = weights[:, idx].view(B, 1, 1, 1)
            weighted_mods.append(m * w)

        # === 3) Concatenate and fuse to feat_dim ===
        concat = torch.cat(weighted_mods, dim=1)  # [B, C*N, H, W]
        fused = self.fuse_conv(concat)             # [B, C, H, W]

        # === 4) Optional Cross-Attention (global) ===
        if self.use_cross_attn:
            # flatten spatial -> seq
            B, C, H, W = fused.shape
            seq_len = H * W
            tokens = fused.view(B, C, seq_len).permute(0, 2, 1)  # [B, S, C] batch_first
            # self-attention (could be cross-attn across modalities but here operate on fused tokens)
            attn_out, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
            # add & norm
            tokens = self.attn_norm1(tokens + attn_out)
            # FFN
            ffn = self.attn_ffn(tokens)
            tokens = self.attn_norm2(tokens + ffn)
            fused = tokens.permute(0, 2, 1).view(B, C, H, W)

        # === 5) Channel attention (SE-style) ===
        ca = self.ca(fused)  # [B, C, 1, 1]
        fused = fused * ca

        # === 6) Spatial attention (CBAM) ===
        avg_pool = fused.mean(dim=1, keepdim=True)  # [B,1,H,W]
        max_pool, _ = fused.max(dim=1, keepdim=True)  # [B,1,H,W]
        sa_input = torch.cat([avg_pool, max_pool], dim=1)  # [B,2,H,W]
        sa = self.spatial_sigmoid(self.spatial_conv(sa_input))  # [B,1,H,W]
        fused = fused * sa

        return fused


import torch
import torch.nn as nn

class FiLMFusionWithAttention(nn.Module):
    """
    多模态特征融合模块：
    1. FiLM 跨模态调制
    2. 自注意力增强 (通道+空间)
    """
    def __init__(self, num_feat):
        super().__init__()
        # FiLM 生成器
        self.gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat * 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 融合后的压缩层
        self.out_conv = nn.Conv2d(num_feat * 3, num_feat, kernel_size=1)

        # Attention 子模块（CBAM风格）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 8, num_feat, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def film(self, x, cond):
        """使用 cond 特征调制 x"""
        gamma_beta = self.gen(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma * x + beta

    def forward(self, f1, f2, f3):
        # === Step 1: FiLM 跨模态调制 ===
        f1_mod = self.film(f1, f2)
        f2_mod = self.film(f2, f3)
        f3_mod = self.film(f3, f1)

        fused = torch.cat([f1_mod, f2_mod, f3_mod], dim=1)
        fused = self.out_conv(fused)

        # === Step 2: 通道注意力 ===
        ca = self.channel_att(fused)
        fused = fused * ca

        # === Step 3: 空间注意力 ===
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa_input)
        fused = fused * sa

        return fused


if __name__ == '__main__':
    # 假设 content_encoder1/2/3 输出 [B,64,32,32]
    enc_dim = 64
    fusion = FiLMFusion(feat_dim=enc_dim, use_cross_attn=True, num_heads=4).cuda()

    # fake batch:
    batch = {
        'T1': torch.randn(2, 64, 32, 32).cuda(),
        'T2': torch.randn(2, 64, 32, 32).cuda(),
        'MRA': torch.randn(2, 64, 32, 32).cuda(),
    }

    c_f1 = batch['T1']
    c_f2 = batch['T2']
    c_f3 = batch['MRA']

    fused = fusion([c_f1, c_f2, c_f3])
    print(fused.shape)  # -> [2, 64, 32, 32]
