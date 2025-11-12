# main.py 或你的脚本最上方
import warnings
from tqdm import tqdm
import os
import yaml
import time
# import visdom

from network.plmodel import MyPlModel, SaveCheck
from network.m2pmodel import M2PModel
from utils.util import dict_as_namespace
from utils.losses import UncertaintyLoss, MIPloss
from data.m2p_dataset import M2PDataset

from lightning.pytorch.loggers import CSVLogger
import pytorch_lightning as pl

import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from torch import optim

# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
torch.multiprocessing.set_sharing_strategy('file_system')


with open('./configs/options.yaml', 'r') as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

def init(opt):
    """
    初始化
    """
    os.makedirs(opt.checkpoints + '/encoder/', exist_ok=True) # encoder权重
    os.makedirs(opt.checkpoints + '/decoder/', exist_ok=True) # decoder权重


def train(options):
    torch.autograd.set_detect_anomaly(True)
    opt = dict_as_namespace(options)
    root = opt.data.niigz_root

    # load dataset
    train_dataset = M2PDataset('train', opt.data.use_slice, root, opt.data.use_modality)
    val_dataset = M2PDataset('valid', opt.data.use_slice, root, opt.data.use_modality)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=opt.train.batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=True,
                                persistent_workers=True)

    # other
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = M2PModel(opt).to(device)
    uncertainty_loss_fn = UncertaintyLoss(lambda_mean=1.0, lambda_nll=1.0)
    mlp_loss_fn = MIPloss(opt)

    # -----------------------
    # 4) 优化器
    # -----------------------
    # 主网络 (encoder/fusion/decoder)
    params_main = list(model.content_encoder1.parameters()) + \
                  list(model.content_encoder2.parameters()) + \
                  list(model.content_encoder3.parameters()) + \
                  list(model.style_encoder1.parameters()) + \
                  list(model.style_encoder2.parameters()) + \
                  list(model.style_encoder3.parameters()) + \
                  list(model.content_fusion.parameters()) + \
                  list(model.style_fusion.parameters()) + \
                  list(model.target_feature.parameters()) + \
                  list(model.decoder.parameters())

    optimizer_main = optim.Adam(params_main, lr=opt.train.lr, weight_decay=getattr(opt.train, "weight_decay", 0.0))


    num_epochs = opt.train.epochs
    lambda_bridge = getattr(opt.train, "lambda_bridge", 1.0)

    # --- Visdom 初始化 ---
    # viz = visdom.Visdom(env='train_loss')
    # loss_win = viz.line(
    #     X=torch.zeros((1,)),
    #     Y=torch.zeros((1, 5)),
    #     opts=dict(
    #         legend=['Total Main', 'Bridge G', 'Bridge D', 'Bridge E', 'Loss F'],
    #         xlabel='Iteration',
    #         ylabel='Loss',
    #         title='Training Loss Curve (Batch level)',
    #         showlegend=True
    #     )
    # )

    global_iter = 0  # 全局迭代计数，用于 X 轴显示 batch

    for epoch in range(opt.train.start, num_epochs + 1):
        model.train()
        epoch_start_time = time.time()
        # 累计损失
        total_loss_main = 0.0
        total_loss_G_bridge = 0.0
        total_loss_D = 0.0
        total_loss_E = 0.0
        total_loss_F = 0.0

        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")

        for i, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if epoch == opt.train.start and i == 0:
                model.initialize_bridge(batch)
                print("✅ bridge / netF data-dependent initialization done")

            p0, c, rho, sigma2 = model.forward(batch)
            gt_p0 = batch.get("PA")
            if gt_p0 is None:
                raise RuntimeError("batch 中没有 PA")
            loss_main, loss_mean, loss_nll = uncertainty_loss_fn(p0, c, rho, sigma2, gt_p0, gt_p0, gt_p0)
            mlp_loss = mlp_loss_fn(p0, batch)
            total_loss = loss_main + mlp_loss

            for name, t in zip(['p0', 'c', 'rho', 'sigma2'], [p0, c, rho, sigma2]):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    print(f"⚠️ NaN or Inf detected in {name}")

            # optimizer_main.zero_grad()
            # total_loss.backward()
            # optimizer_main.step()

            # bridge_losses = model.bridge.optimize_parameters()
            # detach 输出给 bridge，防止共享计算图
            # bridge_losses = model.bridge.optimize_parameters()
            # loss_D = bridge_losses.get("loss_D", torch.tensor(0.0, device=device))
            # loss_E = bridge_losses.get("loss_E", torch.tensor(0.0, device=device))
            # loss_G_bridge = bridge_losses.get("loss_G", torch.tensor(0.0, device=device))

            bridge_losses = model.bridge.optimize_parameters(backward=True)
            loss_D = bridge_losses.get("loss_D", torch.tensor(0.0, device=device))
            loss_E = bridge_losses.get("loss_E", torch.tensor(0.0, device=device))
            loss_G_bridge = bridge_losses.get("loss_G", torch.tensor(0.0, device=device))

            # 联合总损失
            total_loss = loss_main + mlp_loss \
                         + lambda_bridge * (bridge_losses["loss_G"] + bridge_losses["loss_E"] + bridge_losses["loss_D"])

            optimizer_main.zero_grad()
            model.bridge.optimizer_G.zero_grad()
            model.bridge.optimizer_D.zero_grad()
            model.bridge.optimizer_E.zero_grad()

            total_loss.backward()  # ✅ 一次反传搞定所有参数的梯度

            optimizer_main.step()
            model.bridge.optimizer_G.step()
            model.bridge.optimizer_D.step()
            model.bridge.optimizer_E.step()

            total_loss_main += total_loss.item()
            total_loss_G_bridge += loss_G_bridge.item()
            total_loss_D += loss_D.item()
            total_loss_E += loss_E.item()

            # --- Visdom batch-level 更新 ---
            viz.line(
                X=torch.ones((1, 5)) * global_iter,
                Y=torch.tensor([[total_loss_main / (i + 1),
                                 total_loss_G_bridge / (i + 1),
                                 total_loss_D / (i + 1),
                                 total_loss_E / (i + 1),
                                 total_loss_F / (i + 1)]]),
                win=loss_win,
                update='append'
            )
            global_iter += 1

            loop.set_postfix({
                "loss_main": total_loss_main / (i + 1),
                "loss_G": total_loss_G_bridge / (i + 1),
                "loss_D": total_loss_D / (i + 1),
                "loss_E": total_loss_E / (i + 1),
                "loss_F": total_loss_F / (i + 1)
            })

        print(f"Epoch {epoch} finished | Avg Loss_main: {total_loss_main / len(train_dataloader):.4f} "
              f"| Avg Loss_G: {total_loss_G_bridge / len(train_dataloader):.4f} "
              f"| Avg Loss_D: {total_loss_D / len(train_dataloader):.4f} "
              f"| Avg Loss_E: {total_loss_E / len(train_dataloader):.4f} "
              f"| Avg Loss_F: {total_loss_F / len(train_dataloader):.4f} "
              f"| Time: {time.time() - epoch_start_time:.1f}s")



if __name__ == '__main__':
    train(options)

