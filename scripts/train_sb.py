import os
import argparse
import sys
from pathlib import Path

# 将项目根目录（MySynthesisCode）加入Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from models.vqvae import VQVAE
from models.discriminator import BoundaryDisc
from models.score_net import LatentScoreNet, conditioning_dropout
from utils.dataset import ImgDataset, PairedSimDataset
from utils.schedule import linear_beta_schedule, score_matching_loss_latent, q_sample_latent
from utils.losses import physics_consistency_loss_pixels, structure_loss
from config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri_root', type=str, required=True)
    parser.add_argument('--pa_root', type=str, required=True)
    parser.add_argument('--paired_sim_root', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='../outputs/checkpoints_cond_sb')
    parser.add_argument('--batch', type=int, default=cfg.batch_size)
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--time_steps', type=int, default=cfg.time_steps)
    parser.add_argument('--sample_steps', type=int, default=cfg.sample_steps)
    parser.add_argument('--cond_dropout', type=float, default=0.1)
    return parser.parse_args()


def build_loaders(mri_root, pa_root, batch):
    mri_ds = ImgDataset(mri_root)
    pa_ds = ImgDataset(pa_root)
    mri_loader = DataLoader(mri_ds, batch_size=batch, shuffle=True, drop_last=True)
    pa_loader = DataLoader(pa_ds, batch_size=batch, shuffle=True, drop_last=True)
    return mri_loader, pa_loader


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    vqvae = VQVAE().to(device)
    score_net = LatentScoreNet(in_ch=cfg.latent_dim).to(device)
    disc = BoundaryDisc(in_ch=cfg.channels).to(device)

    optim_vq = torch.optim.Adam(vqvae.parameters(), lr=args.lr)
    optim_score = torch.optim.Adam(score_net.parameters(), lr=args.lr)
    optim_d = torch.optim.Adam(disc.parameters(), lr=args.lr)

    mri_loader, pa_loader = build_loaders(args.mri_root, args.pa_root, args.batch)
    paired_loader = None
    if args.paired_sim_root:
        paired_loader = DataLoader(PairedSimDataset(args.paired_sim_root), batch_size=args.batch, shuffle=True,
                                   drop_last=True)
        paired_iter = iter(paired_loader)

    betas = linear_beta_schedule(args.time_steps)
    bce = nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(zip(mri_loader, pa_loader), total=min(len(mri_loader), len(pa_loader)))
        for mri_batch, pa_batch in pbar:
            mri_batch = mri_batch.to(device)
            pa_batch = pa_batch.to(device)
            B = mri_batch.size(0)

            # Train VQ-VAE (reconstruction) for stability
            vqvae.train()
            recon_mri, qz_mri, qloss_mri = vqvae(mri_batch)
            recon_pa, qz_pa, qloss_pa = vqvae(pa_batch)
            loss_vq = F.mse_loss(recon_mri, mri_batch) + F.mse_loss(recon_pa, pa_batch) + (qloss_mri + qloss_pa)
            optim_vq.zero_grad();
            loss_vq.backward();
            optim_vq.step()

            # Encode latents (use quantized latents as z0)
            vqvae.eval()
            with torch.no_grad():
                z_mri = vqvae.enc(mri_batch)
                z_pa = vqvae.enc(pa_batch)
                qz_mri, _ = vqvae.quant(z_mri)
                qz_pa, _ = vqvae.quant(z_pa)

            # Classifier-free guidance: randomly drop conditioning
            cond_qz = conditioning_dropout(qz_mri, drop_prob=args.cond_dropout)

            # Score matching losses
            t = torch.rand(B, device=device)
            Ls_mri = score_matching_loss_latent(score_net, qz_mri, t, cond_qz)
            Ls_pa = score_matching_loss_latent(score_net, qz_pa, t, cond_qz)
            L_score = 0.5 * (Ls_mri + Ls_pa)

            # Physics loss (if paired sim available)
            if paired_loader is not None:
                try:
                    mri_sim, pa_sim_stats = next(paired_iter)
                except StopIteration:
                    paired_iter = iter(paired_loader)
                    mri_sim, pa_sim_stats = next(paired_iter)
                mri_sim = mri_sim.to(device)
                with torch.no_grad():
                    z_mri_sim = vqvae.enc(mri_sim)
                    qz_mri_sim, _ = vqvae.quant(z_mri_sim)
                # Simple unconditional sampling and decode (conditional sampler can be added later)
                gen_dec, _ = sample_latent_uncond(score_net, vqvae, args.sample_steps, B, device)
                L_phys = physics_consistency_loss_pixels(gen_dec, pa_sim_stats)
            else:
                L_phys = torch.tensor(0., device=device)

            # Structure loss: encourage anatomical consistency in latent
            with torch.no_grad():
                t_proxy = torch.zeros(B, device=device)
                noise = torch.randn_like(qz_mri)
                zt = q_sample_latent(qz_mri, torch.zeros(B, dtype=torch.long, device=device), noise)
                gen_proxy = score_net(zt, t_proxy, cond_qz)
            L_struct = F.mse_loss(gen_proxy, qz_mri)

            loss = L_score + cfg.physics_loss_weight * L_phys + cfg.structure_loss_weight * L_struct
            optim_score.zero_grad();
            loss.backward();
            optim_score.step()

            # Discriminator updates (boundary matching at t=1)
            disc.train()
            real_logits = disc(pa_batch)
            real_targets = torch.ones_like(real_logits)
            loss_d_real = bce(real_logits, real_targets)

            with torch.no_grad():
                fake_dec, _ = sample_latent_uncond(score_net, vqvae, args.sample_steps, B, device)
            fake_logits = disc(fake_dec)
            fake_targets = torch.zeros_like(fake_logits)
            loss_d_fake = bce(fake_logits, fake_targets)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            optim_d.zero_grad();
            loss_d.backward();
            optim_d.step()

            # Generator boundary fooling
            fake_dec2, _ = sample_latent_uncond(score_net, vqvae, args.sample_steps, B, device)
            loss_g_bdy = bce(disc(fake_dec2), torch.ones_like(fake_logits))
            optim_score.zero_grad();
            loss_g_bdy.backward();
            optim_score.step()

            if step % 100 == 0:
                pbar.set_description(
                    f"Ep{epoch} Step{step} Loss{loss.item():.4f} Ls{L_score.item():.4f} Lphys{L_phys.item():.4f}")

            if step % 500 == 0:
                torch.save({'vqvae': vqvae.state_dict(), 'score': score_net.state_dict(), 'disc': disc.state_dict()},
                           os.path.join(args.save_dir, f'model_step{step}.pth'))

            step += 1

    torch.save({'vqvae': vqvae.state_dict(), 'score': score_net.state_dict(), 'disc': disc.state_dict()},
               os.path.join(args.save_dir, 'model_final.pth'))


# helper unconditional latent sampler used above
@torch.no_grad()
def sample_latent_uncond(score_net, vqvae, steps, batch, device):
    z = torch.randn(batch, cfg.latent_dim, cfg.z_h, cfg.z_h, device=device)
    for i in reversed(range(steps)):
        t = torch.full((batch,), float(i) / float(steps - 1), device=device)
        s = score_net(z, t, None)
        beta = linear_beta_schedule(cfg.time_steps)[i].to(device)
        z = z + 0.5 * beta * s
        if i > 0:
            z = z + torch.sqrt(beta) * torch.randn_like(z)
    recon = vqvae.dec(z)
    return recon, z


if __name__ == '__main__':
    args = parse_args()
    train(args)
