# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：train_vqvae.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 20:56 
"""
import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
from pathlib import Path
import time
import warnings

# 将项目根目录（MySynthesisCode）加入Python路径
sys.path.append(str(Path(__file__).parent.parent))  # 若当前文件在 scripts/ 下，则 parent.parent 为根目录

import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn import L1Loss

from monai.networks.nets import VQVAE
from monai.utils import set_determinism, ensure_tuple

from utils.dataset import get_dataloader

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='VQ-VAE')
    parser.add_argument('--log-folder', type=str, default='vqvae',
                        help='name of the output folder (default: vqvae)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='name of the output folder (default: outputs)')
    parser.add_argument("--training_ids", default="./outputs/ids/train.tsv", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", default="./outputs/ids/validation.tsv",
                        help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of loader workers")

    return parser.parse_args()


def main_oral_vqvae(args):
    # 使用自己写的代码：https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/vqvae.py#L64
    # Create logs and models folder if they don't exist
    # 获取父亲的父亲目录（项目根目录）
    root_dir = Path(__file__).parent.parent

    # logs 路径 = 根目录 / logs / output-folder
    log_dir = root_dir / "logs" / args.log_folder
    log_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(str(log_dir))

    cache_dir = root_dir / args.output_dir / "cached_data_vqvae"
    cache_dir.mkdir(exist_ok=True)
    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=root_dir / args.training_ids,
        validation_ids=root_dir / args.validation_ids,
        num_workers=args.num_workers,
        model_type="vqvae",
    )

    ##############################
    # # Fixed images for Tensorboard
    # fixed_images, _ = next(iter(test_loader))
    # fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    # writer.add_image('original', fixed_grid, 0)

    # from models.vqvae import VQVAE
    # device = torch.device('0' if torch.cuda.is_available() else 'cpu')
    # model = VQVAE(num_channels, args.hidden_size, args.k).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    # reconstruction = generate_samples(fixed_images, model, args)
    # grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    # writer.add_image('reconstruction', grid, 0)

    # best_loss = -1.
    # for epoch in range(args.num_epochs):
    #     train(train_loader, model, optimizer, args, writer)
    #     loss, _ = test(valid_loader, model, args, writer)
    #
    #     reconstruction = generate_samples(fixed_images, model, args)
    #     grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    #     writer.add_image('reconstruction', grid, epoch + 1)
    #
    #     if (epoch == 0) or (loss < best_loss):
    #         best_loss = loss
    #         with open('{0}/best.pt'.format(save_filename), 'wb') as f:
    #             torch.save(model.state_dict(), f)
    #     with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
    #         torch.save(model.state_dict(), f)
    # return 0


def main(args):
    # 获取父亲的父亲目录（项目根目录）
    root_dir = Path(__file__).parent.parent

    # logs 路径 = 根目录 / logs / output-folder
    log_dir = root_dir / "logs" / args.log_folder
    log_dir.mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(str(log_dir))
    output_dir = root_dir / args.output_dir

    cache_dir = output_dir / "cached_data_vqvae"
    cache_dir.mkdir(exist_ok=True)
    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=root_dir / args.training_ids,
        validation_ids=root_dir / args.validation_ids,
        num_workers=args.num_workers,
        model_type="vqvae",
    )

    set_determinism(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 64),  # 原为 (256, 256)
        num_res_channels=64,  # 原为 256
        num_res_layers=1,  # 原为 2
        downsample_parameters=((2, 2, 1, 1), (2, 2, 1, 1)),  # 减小 stride 和 kernel
        upsample_parameters=((2, 2, 1, 1, 0), (2, 2, 1, 1, 0)),
        num_embeddings=128,  # 原为 256
        embedding_dim=16,  # 原为 32
    ).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    l1_loss = L1Loss()

    ##################
    # training
    ##################
    max_epochs = 10
    val_interval = 2
    epoch_recon_loss_list = []
    epoch_quant_loss_list = []
    val_recon_epoch_loss_list = []
    best_val_loss = float('inf')

    checkpoint_dir = output_dir / 'vqvae_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_save_path = output_dir / 'best_vqvae.pth'

    total_start = time.time()
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)

            # model outputs reconstruction and the quantization error
            reconstruction, quantization_loss = model(images=images)

            recons_loss = l1_loss(reconstruction.float(), images.float())

            loss = recons_loss + quantization_loss

            loss.backward()
            optimizer.step()

            epoch_loss += recons_loss.item()

            progress_bar.set_postfix(
                {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    reconstruction, quantization_loss = model(images=images)

                    recons_loss = l1_loss(reconstruction.float(), images.float())
                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

            ## 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_save_path)

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)

    total_time = time.time() - total_start

    print(f"train completed, total time: {total_time}.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
