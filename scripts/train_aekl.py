# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：train_aekl.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/9/23 17:50 
"""
""" 带KL正则化的自动编码器训练脚本 """
import argparse
import warnings
from pathlib import Path

import os
os.environ["TORCH_HOME"] = "D:\postgraduate\diffusion_PA\code\MySynthesisCode\outputs\cache"


import torch
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_aekl
from util import get_dataloader  # 移除log_mlflow导入

from utils.path import PROJECT_ROOT

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", default="./outputs/run_aekl", help="Location of model to resume.")
    parser.add_argument("--training_ids", default="./outputs/ids/train.tsv", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", default="./outputs/ids/validation.tsv",
                        help="Location of file with validation ids.")
    parser.add_argument("--config_file", default="./configs/aekl.yaml",
                        help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch when the adversarial training starts.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    # parser.add_argument("--experiment", help="Mlflow experiment name.")
    return parser.parse_args()


def main(args):
    # 初始化部分保持不变
    set_determinism(seed=args.seed)
    print_config()

    output_dir = PROJECT_ROOT / 'outputs/runs/'
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = PROJECT_ROOT / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"运行目录: {run_dir}")
    print("运行参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # 保留TensorBoard记录器
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    # 数据加载部分保持不变
    cache_dir = output_dir / "cached_data_aekl"
    cache_dir.mkdir(exist_ok=True)
    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=PROJECT_ROOT / args.training_ids,
        validation_ids=PROJECT_ROOT / args.validation_ids,
        num_workers=args.num_workers,
        model_type="autoencoder",
    )

    # 模型初始化保持不变
    config = OmegaConf.load(PROJECT_ROOT / args.config_file)
    print(f"config['edit']['run_dir']:${config['edit']['run_dir']}\n\n")
    model = AutoencoderKL(**config["stage1"]["params"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])

    # 设备设置保持不变
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)
    model.to(device)
    perceptual_loss.to(device)
    discriminator.to(device)

    # 优化器保持不变
    optimizer_g = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])

    # 检查点恢复保持不变
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print("检测到检查点，恢复训练...")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

    # 训练过程保持不变（已移除MLflow依赖）
    print("开始训练...")
    val_loss = train_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        kl_weight=config["stage1"]["kl_weight"],
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.adv_start,
    )

    print(f"训练完成，最终验证损失: {val_loss:.4f}")


if __name__ == "__main__":

    args = parse_args()
    main(args)