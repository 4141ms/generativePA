import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1. VQ-VAE 模块
# -----------------------------
class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden=64, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden*2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden*2, latent_dim, 3, 1, 1),
        )

    def forward(self, x):
        z_e = self.conv(x)
        return z_e  # [B, latent_dim, H/4, W/4]


class VQVAEVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: [B,C,H,W]
        B,C,H,W = z.shape
        z_flat = z.permute(0,2,3,1).contiguous().view(-1,C)
        # 计算最近 embedding
        d = torch.cdist(z_flat.unsqueeze(0), self.embedding.weight.unsqueeze(0))[0]  # [N, num_embeddings]
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(B,H,W,C).permute(0,3,1,2)
        # commitment loss
        loss = F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())
        # straight-through
        z_q = z + (z_q - z).detach()
        return z_q, loss


class VQVAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden=128, out_channels=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden*2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden*2, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, out_channels, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, z):
        x_rec = self.deconv(z)
        return x_rec

# -----------------------------
# 2. Latent SB 模块
# -----------------------------
class LatentSB(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, z_mri, cond):
        # z_mri: [B,C,H,W] -> flatten
        B,C,H,W = z_mri.shape
        z_flat = z_mri.view(B, -1)
        z_in = torch.cat([z_flat, cond], dim=1)
        z_out = self.net(z_in)
        # reshape back to latent map
        z_out = z_out.view(B,C,H,W)
        return z_out

# -----------------------------
# 3. 物理一致性损失
# -----------------------------
def physics_loss(pa_pred, vessel_mask, sO2_map, mu_a_oxy=0.1, mu_a_deoxy=0.05, gamma=1.0):
    mu_a = vessel_mask * (sO2_map * mu_a_oxy + (1 - sO2_map) * mu_a_deoxy)
    phi = torch.exp(-mu_a)
    pa_ref = gamma * mu_a * phi
    loss = F.l1_loss(pa_pred, pa_ref)
    return loss

# -----------------------------
# 4. Minimal forward pass
# -----------------------------
B,C,H,W = 2,1,224,224
x_mri = torch.rand(B,C,H,W)
vessel_mask = (torch.rand(B,1,H,W) > 0.7).float()
sO2_map = torch.rand(B,1,H,W) * vessel_mask
cond = torch.cat([sO2_map.mean(dim=(2,3)), vessel_mask.mean(dim=(2,3))], dim=1)  # batch-level cond

# init models
latent_dim = 128
encoder = VQVAEEncoder(in_channels=C, latent_dim=latent_dim)
quantizer = VQVAEVectorQuantizer(num_embeddings=512, embedding_dim=latent_dim)
sb = LatentSB(latent_dim=latent_dim, cond_dim=2)
decoder = VQVAEDecoder(latent_dim=latent_dim, out_channels=C)

# forward
z_e = encoder(x_mri)
z_q, loss_commit = quantizer(z_e)
z_pa = sb(z_q, cond)
pa_pred = decoder(z_pa)

# compute physics loss
loss_phys = physics_loss(pa_pred, vessel_mask, sO2_map)
total_loss = loss_commit + loss_phys

print("Commitment Loss:", loss_commit.item())
print("Physics Loss:", loss_phys.item())
print("Total Loss:", total_loss.item())
