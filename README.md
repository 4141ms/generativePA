# README: MRI -> PA Image Synthesis using Conditional Schrödinger Bridge (SB)

This README explains how to run the training and conditional sampling pipeline for converting MRI to PA images using the latent-space Schrödinger Bridge model implemented in PyTorch.

---

## 1. Project Structure

```
project_root/
│
├─ models/
│   ├─ vqvae.py           # VQ-VAE modules (Encoder, Decoder, Quantizer)
│   ├─ discriminator.py    # Boundary Discriminator
│   └─ score_net.py        # Latent-space Score Network (conditional)
│
├─ utils/
│   ├─ dataset.py          # MRI / PA / Paired Sim datasets
│   ├─ schedule.py         # Beta schedule, q_sample_latent, score-matching loss
│   └─ losses.py           # physics_consistency_loss, structure_loss
│
├─ scripts/
│   ├─ train_sb.py         # Training script for SB
│   ├─ sample.py           # Unconditional latent-space sampling
│   └─ cond_sample.py      # (Optional) Conditional sampling CLI
│
├─ config.py               # Configuration (latent dim, channels, lr, etc.)
└─ README.md               # This file
```

---

## 2. Environment

* Python >=3.9
* PyTorch >=2.0
* torchvision >=0.15
* tqdm
* PIL / Pillow

Install dependencies (example):

```bash
pip install torch torchvision tqdm pillow
```

---

## 3. Training

### 3.1 Prepare datasets

* `mri_root/`: folder of MRI images (png/jpg)
* `pa_root/`: folder of real PA images (optional if only training score network with unpaired data)
* `paired_sim_root/`: optional folder containing paired MRI -> PA simulated stats

### 3.2 Run training

```bash
python scripts/train_sb.py \
    --mri_root /path/to/mri \
    --pa_root /path/to/pa \
    --paired_sim_root /path/to/paired_sim \
    --save_dir ./checkpoints \
    --batch 8 \
    --epochs 200 \
    --lr 2e-4
```

* Checkpoints are saved every 500 steps.
* The final model is saved as `model_final.pth` in `save_dir`.
* Training uses classifier-free guidance dropout; adjust `--cond_dropout` if needed.

### 3.3 Notes

* VQ-VAE is trained jointly to stabilize latent representation.
* Physics loss and structure loss are optionally computed if paired data is provided.
* Discriminator is used to improve boundary fidelity.

---

## 4. Conditional Sampling (MRI -> PA)

### 4.1 Prepare MRI image

* Load a grayscale MRI image and resize to `cfg.img_size`.
* Use the VQ-VAE encoder to obtain latent `z_mri`.

### 4.2 Run conditional sampling

```python
from models.vqvae import VQVAE
from models.score_net import LatentScoreNet, sample_conditional
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vqvae = VQVAE().to(device)
score_net = LatentScoreNet().to(device)

ck = torch.load('checkpoints/model_final.pth', map_location=device)
vqvae.load_state_dict(ck['vqvae'])
score_net.load_state_dict(ck['score'])

# MRI latent: (B, C, H, W)
z_mri = vqvae.enc(mri_tensor.to(device))
z_mri_q, _ = vqvae.quant(z_mri)

# Generate PA
recon_pa, z_pa = sample_conditional(score_net, vqvae, z_mri_q, steps=200, guidance_scale=2.0, device=device)
```

* `guidance_scale >1.0` amplifies conditioning.
* Output `recon_pa` is in [-1,1]; rescale to [0,255] for saving as PNG.

### 4.3 Optional CLI (if cond_sample.py exists)

```bash
python scripts/cond_sample.py --ck checkpoints/model_final.pth --mri path/to/mri.png --out ./samples --guidance 2.0
```

---

## 5. Tips & Best Practices

* Start training with smaller `batch` and `latent_dim` to debug.
* For larger images or 3D volumes, you need to modify VQ-VAE and score network to 3D convolutions.
* Use mixed precision (`torch.cuda.amp`) to reduce GPU memory usage.
* Experiment with `time_steps` and `sample_steps` for better trade-off between quality and sampling speed.
* Monitor `physics_loss` and `structure_loss` for conditional consistency.

---

## 6. References

* Schrödinger Bridge for generative modeling in latent space.
* VQ-VAE: "Neural Discrete Representation Learning" (van den Oord et al.)
* Classifier-Free Guidance (Ho & Salimans, 2022)
* Physics-consistency loss: enforce simulated PA statistics

---

This README provides a concise guide to training and conditional inference for MRI -> PA image synthesis using the latent-space SB framework.
