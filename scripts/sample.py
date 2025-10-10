# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：sample.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:29 
"""
## scripts/sample.py
# scripts/sample.py

import sys
from pathlib import Path

# 将项目根目录（MySynthesisCode）加入Python路径
sys.path.append(str(Path(__file__).parent.parent))  # 若当前文件在 scripts/ 下，则 parent.parent 为根目录

import argparse
import torch
from pathlib import Path
from models.vqvae import VQVAE
from models.score_net import LatentScoreNet
from utils.schedule import linear_beta_schedule
from config import cfg


def sample_from_checkpoint(checkpoint, out_dir, batch=4, steps=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae = VQVAE().to(device)
    score = LatentScoreNet().to(device)
    ck = torch.load(checkpoint, map_location=device)
    if 'vqvae' in ck:
        vqvae.load_state_dict(ck['vqvae'])
    if 'score' in ck:
        score.load_state_dict(ck['score'])
    vqvae.eval();
    score.eval()
    z = torch.randn(batch, cfg.latent_dim, cfg.z_h, cfg.z_h, device=device)
    # simple reverse sampling (Euler-like)
    betas = linear_beta_schedule(cfg.time_steps)
    for i in reversed(range(steps)):
        t = torch.full((batch,), float(i) / float(steps - 1), device=device)
        s = score(z, t, None)
        beta = betas[i].to(device)
        z = z + 0.5 * beta * s
        if i > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)
    reco = vqvae.dec(z)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(reco.size(0)):
        # denormalize from [-1,1] to [0,255]
        img = ((reco[i].clamp(-1, 1) + 1) / 2.0).mul(255).cpu().numpy().astype('uint8').squeeze()
        from PIL import Image
        Image.fromarray(img).save(out_dir / f'sample_{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ck', required=True)
    parser.add_argument('--out', default='./samples')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    sample_from_checkpoint(args.ck, args.out, args.batch, args.steps)
