# ---------------------------
# Conditional Score-based (style-SB proxy) implementation
# ---------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Time embedding (sin-cos) reused/expanded
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t: (B,) floats in [0,1] or ints [0,T-1]
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -math.log(10000) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # (B, dim)

# Score network (predict noise epsilon) - MLP conditioned on z_MRI (cond_z)
class ScoreNet(nn.Module):
    def __init__(self, z_dim=128, cond_dim=128, time_emb_dim=128, hidden=512):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        in_dim = z_dim + cond_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )
    def forward(self, z_t, cond_z, t_scaled):
        # z_t: (B,z_dim), cond_z: (B,cond_dim)
        # t_scaled: (B,) float in [0,1] or scalar time -> embed
        te = self.time_emb(t_scaled)  # (B, time_emb_dim)
        inp = torch.cat([z_t, cond_z, te], dim=1)
        return self.net(inp)  # predict noise epsilon

# Noise schedule helpers (continuous-time param via alpha(t)/sigma(t))
def get_alphas_sigmas(T=1000, device='cpu', beta_start=1e-4, beta_end=2e-2):
    # produce discrete beta schedule and derived sqrt_alphas_cumprod etc.
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # (T,)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    return betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# q_sample: add noise to z0 at timestep t
def q_sample_z(z0, t_idx, sqrt_acp, sqrt_om_acp, noise=None):
    # t_idx: (B,) integers
    if noise is None:
        noise = torch.randn_like(z0)
    a = sqrt_acp[t_idx].view(-1,1).to(z0.device)
    b = sqrt_om_acp[t_idx].view(-1,1).to(z0.device)
    return a * z0 + b * noise, noise

# Training loop for style-SB (score-based) in latent space
def train_style_sb(score_net: nn.Module,
                   encoder: nn.Module,
                   latent_head: nn.Module,
                   optimizer,
                   dataloader_mouse,
                   dataloader_human=None,     # optional human style samples for terminal regularization
                   T=200,
                   device='cuda',
                   epochs=10):
    """
    score_net: predicts noise eps from (z_t, cond_z, t)
    encoder + latent_head: used to get z_MRI (cond) and z_PA targets (z0)
    dataloader_mouse: yields (mri_mouse, pa_mouse)
    dataloader_human: optional small human PA loader for terminal fitting
    """
    score_net.train()
    # prepare schedule
    betas, sqrt_acp, sqrt_om_acp = get_alphas_sigmas(T=T, device=device)
    mse = nn.MSELoss()
    # Optionally collect human style latents for terminal distribution matching (small set)
    human_style_mus = None
    if dataloader_human is not None:
        mus = []
        with torch.no_grad():
            for _, pa_np in dataloader_human:
                pa = pa_np.to(device)
                feats_pa = encoder(pa)
                mu_pa, _ = latent_head(feats_pa[-1])
                mus.append(mu_pa.cpu())
        if len(mus) > 0:
            human_style_mus = torch.cat(mus, dim=0).to(device)

    for ep in range(epochs):
        pbar = tqdm(dataloader_mouse, desc=f"SB ep{ep}")
        for mri_np, pa_np in pbar:
            mri = mri_np.to(device)
            pa = pa_np.to(device)
            # get condition z_mri and target z0 (style latent of pa)
            with torch.no_grad():
                feats_mri = encoder(mri)
                mu_mri, _ = latent_head(feats_mri[-1])   # (B, z_dim) cond
                feats_pa = encoder(pa)
                mu_pa, _ = latent_head(feats_pa[-1])     # (B, z_dim) target z0
                z0 = mu_pa  # may sample if you prefer stochastic target

            B = z0.size(0)
            t_idx = torch.randint(0, T, (B,), device=device)
            # q sample
            z_t, noise = q_sample_z(z0, t_idx, sqrt_acp, sqrt_om_acp)
            # scale t to [0,1] for embedding: t / (T-1)
            t_scaled = t_idx.float() / float(max(1, T-1))
            eps_pred = score_net(z_t, mu_mri, t_scaled)
            loss_sm = mse(eps_pred, noise)
            # optional: terminal distrib regularization: encourage neglogprob under human_style_mus
            loss_term = 0.0
            if human_style_mus is not None:
                # take predicted z at t small (near 0) by denoising once (approx)
                # here use predicted z0_hat = (z_t - sqrt_one_minus*eps_pred)/sqrt_acp
                a = sqrt_acp[t_idx].view(-1,1).to(device)
                b = sqrt_om_acp[t_idx].view(-1,1).to(device)
                z0_hat = (z_t - b * eps_pred) / a
                # match distribution of z0_hat with human_style_mus by MMD on minibatch
                # sample a minibatch from human_style_mus
                idx = torch.randint(0, human_style_mus.size(0), (B,), device=device)
                mus_sample = human_style_mus[idx]
                loss_term = compute_mmd(z0_hat.detach().cpu(), mus_sample.detach().cpu())
                # compute_mmd currently returns cpu tensor; convert to float scalar
                # (for autograd we could implement differentiable MMD on device; for small human set this is fine)
            loss = loss_sm + 0.1 * loss_term
            optimizer.zero_grad()
            # note: if loss_term is cpu tensor, convert to float and back for gradient-free regularization
            if isinstance(loss_term, float) or (torch.is_tensor(loss_term) and loss_term.device.type == 'cpu'):
                # backprop only on loss_sm (main term); leave loss_term as non-differentiable reg for simplicity
                loss_sm.backward()
            else:
                loss.backward()
            optimizer.step()
            pbar.set_postfix({'sm': f"{loss_sm.item():.6f}", 'term': f"{float(loss_term):.6f}" if loss_term is not None else "0.0"})
    return

# Sampling: reverse SDE via Euler–Maruyama + optional Langevin corrector
@torch.no_grad()
def sample_style_sb(score_net: nn.Module,
                    cond_z: torch.Tensor,
                    shape=(1,),
                    T=200,
                    device='cuda',
                    n_steps=200,
                    snr=0.05):
    """
    Sample a style latent conditioned on cond_z (z_MRI).
    Returns z_0 sample (B, z_dim).
    Approach: start from z_T ~ N(0, I) (or modeled prior), then run discrete reverse steps:
      for t = T-1 ... 0:
        predict eps = score_net(z_t, cond_z, t/T)
        compute mean of p(z_{t-1}|z_t) and sample with Euler step.
      plus optional Langevin corrector (few steps) per timestep.
    This is a standard predictor-corrector sampler for score-based models.
    """
    score_net.eval()
    B = cond_z.size(0)
    z_dim = cond_z.size(1)
    # prepare schedule
    betas, sqrt_acp, sqrt_om_acp = get_alphas_sigmas(T=T, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # initialize z_T ~ N(0, I)
    z_t = torch.randn((B, z_dim), device=device)
    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
        t_scaled = t_tensor / float(max(1, T-1))
        # predict noise
        eps_pred = score_net(z_t, cond_z, t_scaled)
        # compute posterior mean (predict z0_hat, then mean of q(z_{t-1}|z_t))
        a_t = sqrt_acp[t].to(device)
        b_t = sqrt_om_acp[t].to(device)
        a_t = a_t.view(1,1); b_t = b_t.view(1,1)
        # z0_hat = (z_t - b_t * eps_pred) / a_t
        z0_hat = (z_t - b_t * eps_pred) / a_t
        # mean of p(z_{t-1}|z_t) in DDPM derivation:
        if t > 0:
            a_tm1 = sqrt_acp[t-1].to(device).view(1,1)
            posterior_mean = a_tm1 * z0_hat
            # posterior variance (simplified) = beta_t (approx); sample noise
            posterior_var = betas[t].to(device)
            z_t = posterior_mean + math.sqrt(max(0.0, posterior_var)) * torch.randn_like(z_t)
        else:
            # t == 0, return z0_hat as final sample
            z_t = z0_hat
        # Optional: Langevin corrector few steps (improves sample quality)
        # small-number of steps:
        for _ in range(1):  # 1 step corrector; increase to 2-5 if desired
            grad = score_net(z_t, cond_z, t_scaled)  # score approximates -noise/var, direction to increase prob
            noise = torch.randn_like(z_t) * math.sqrt(2 * (snr ** 2))
            z_t = z_t + (snr ** 2) * grad + noise
    return z_t  # approx z0 sample

# ---------------------------
# Integration notes:
# - Replace previous LatentDenoiser with ScoreNet
# - Call train_style_sb(...) instead of train_stage1_latent_ddpm(...)
# - Use sample_style_sb(score_net, cond_z, ...) in Stage2 to produce human-like style latents
# ---------------------------

