import torch
from datetime import datetime
from pathlib import Path
import numpy as np
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImage, ToTensor
from monai.utils import set_determinism
from torch.optim import Adam
from tqdm import tqdm

# 配置参数（从JSON中提取）
config = {
    "bundle_root": ".",
    "model_dir": "./models",
    "output_dir": "./output",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "autoencoder_params": {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 3,
        "channels": [64, 128, 128, 128],
        "num_res_blocks": 2,
        "attention_levels": [False, False, False, False],
    },
    "diffusion_params": {
        "spatial_dims": 3,
        "in_channels": 7,  # 3 (latent) + 4 (conditioning)
        "out_channels": 3,
        "channels": [256, 512, 768],
        "num_res_blocks": 2,
        "attention_levels": [False, True, True],
        "cross_attention_dim": 4,
        "with_conditioning": True,
    },
    "scheduler_params": {
        "beta_start": 0.0015,
        "beta_end": 0.0205,
        "num_train_timesteps": 50,
    },
    "batch_size": 2,
    "epochs": 100,
}

# 初始化目录
Path(config["model_dir"]).mkdir(exist_ok=True)
Path(config["output_dir"]).mkdir(exist_ok=True)

# 设置随机种子
set_determinism(42)

# ----------------------------------------
# 1. 数据加载
# ----------------------------------------
class CustomDataset(Dataset):
    def __init__(self, image_files, conditioning):
        self.image_files = image_files
        self.conditioning = conditioning
        self.transform = Compose([LoadImage(image_only=True), ToTensor()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = self.transform(self.image_files[idx])
        cond = torch.tensor(self.conditioning[idx], dtype=torch.float32)
        return img, cond

# 示例数据（替换为实际数据）
image_files = ["../dataset/ds002868/sub-001/ses-1/anat/sub-001_ses-1_acq-RARE_T2w.nii.gz", "../dataset/ds002868/sub-002/ses-1/anat/sub-002_ses-1_acq-RARE_T2w.nii.gz"]
conditioning = [
    [0.0, 0.1, 0.2, 0.4],  # [gender, age, ventricular_vol, brain_vol]
    [1.0, 0.3, 0.1, 0.3],
]

dataset = CustomDataset(image_files, conditioning)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# ----------------------------------------
# 2. 模型定义
# ----------------------------------------
autoencoder = AutoencoderKL(**config["autoencoder_params"]).to(config["device"])
diffusion = DiffusionModelUNet(**config["diffusion_params"]).to(config["device"])
scheduler = DDIMScheduler(**config["scheduler_params"])

# ----------------------------------------
# 3. 训练逻辑
# ----------------------------------------
def train():
    optimizer_ae = Adam(autoencoder.parameters(), lr=1e-4)
    optimizer_diff = Adam(diffusion.parameters(), lr=1e-4)

    for epoch in range(config["epochs"]):
        autoencoder.train()
        diffusion.train()
        progress = tqdm(dataloader, desc=f"Epoch {epoch}")

        for images, conditions in progress:
            images = images.to(config["device"])
            conditions = conditions.to(config["device"])

            # --- 训练Autoencoder ---
            optimizer_ae.zero_grad()
            latent, _, _ = autoencoder.encode(images)
            recon = autoencoder.decode(latent)
            loss_ae = torch.nn.functional.mse_loss(recon, images)
            loss_ae.backward()
            optimizer_ae.step()

            # --- 训练Diffusion Model ---
            optimizer_diff.zero_grad()
            latent = autoencoder.encode_stochastic(images)[0]
            noise = torch.randn_like(latent)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latent.shape[0],)).to(config["device"])

            noisy = scheduler.add_noise(latent, noise, timesteps)
            pred_noise = diffusion(noisy, timesteps, context=conditions.unsqueeze(1))
            loss_diff = torch.nn.functional.mse_loss(pred_noise, noise)
            loss_diff.backward()
            optimizer_diff.step()

            progress.set_postfix({"AE Loss": loss_ae.item(), "Diff Loss": loss_diff.item()})

        # 保存模型
        if epoch % 10 == 0:
            torch.save(autoencoder.state_dict(), f"{config['model_dir']}/autoencoder_epoch{epoch}.pt")
            torch.save(diffusion.state_dict(), f"{config['model_dir']}/diffusion_epoch{epoch}.pt")

# ----------------------------------------
# 4. 推理生成
# ----------------------------------------
def infer(conditioning):
    autoencoder.eval()
    diffusion.eval()

    # 生成随机噪声
    latent_shape = (1, 3, 20, 28, 20)  # 假设的潜在空间形状
    noise = torch.randn(latent_shape).to(config["device"])
    conditioning = torch.tensor(conditioning).to(config["device"]).unsqueeze(0)

    # DDIM采样
    scheduler.set_timesteps(50)
    latent = noise
    for t in tqdm(scheduler.timesteps):
        pred_noise = diffusion(latent, t, context=conditioning.unsqueeze(1))
        latent = scheduler.step(pred_noise, t, latent).prev_sample

    # 解码生成图像
    with torch.no_grad():
        image = autoencoder.decode(latent)
    
    return image.cpu()

# ----------------------------------------
# 执行训练和推理
# ----------------------------------------
if __name__ == "__main__":
    train()  # 训练模型

    # 示例推理
    sample = infer(conditioning=[0.0, 0.1, 0.2, 0.4])
    print("Generated sample shape:", sample.shape)