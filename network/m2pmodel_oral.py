import warnings

# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.utils import make_grid
from model.BrownianBridge.BrownianBridge import BrownianBridgeModel
from model.UNSB.sb_model import SBModel
from pytorch_lightning.utilities import rank_zero_only

import pytorch_lightning as pl
import torch
from network.VSSM_Encoder import VSSMEncoder
import torch.nn as nn
# class SBModelWrapper(pl.LightningModule):
#     """
#     Lightning wrapper for SBModel (inherits BaseModel).
#     Provides manual optimization support and integrates SBModel optimizers.
#     """
#     def __init__(self, sb_opt):
#         super().__init__()
#         self.UNSB = SBModel(sb_opt)
#         self.automatic_optimization = False  # ✅ 手动优化
#
#         # 获取 SBModel 内部优化器
#         self.optimizer_G = getattr(self.UNSB, 'optimizer_G', None)
#         self.optimizer_D = getattr(self.UNSB, 'optimizer_D', None)
#         self.optimizer_E = getattr(self.UNSB, 'optimizer_E', None)
#         self.optimizer_F = getattr(self.UNSB, 'optimizer_F', None)
#
#     def forward(self, input_A, input_B=None):
#         if input_B is not None:
#             self.UNSB.set_input({'A': input_A, 'B': input_B})
#         else:
#             self.UNSB.set_input({'A': input_A})
#         self.UNSB.forward()
#         return self.UNSB.fake_B  # 默认输出 fake_B
#
#     def training_step(self, batch, batch_idx):
#         opt_G, opt_D, opt_E, opt_F = self.optimizers()
#
#         # 设置输入
#         self.UNSB.set_input(batch)
#         self.UNSB.forward()
#
#         # ------------------- update D -------------------
#         self.UNSB.set_requires_grad(self.UNSB.netD, True)
#         opt_D.zero_grad()
#         loss_D = self.UNSB.compute_D_loss()
#         self.manual_backward(loss_D)
#         opt_D.step()
#
#         # ------------------- update E -------------------
#         self.UNSB.set_requires_grad(self.UNSB.netE, True)
#         opt_E.zero_grad()
#         loss_E = self.UNSB.compute_E_loss()
#         self.manual_backward(loss_E)
#         opt_E.step()
#
#         # ------------------- update G & F -------------------
#         self.UNSB.set_requires_grad([self.UNSB.netD, self.UNSB.netE], False)
#         opt_G.zero_grad()
#         if opt_F: opt_F.zero_grad()
#         loss_G = self.UNSB.compute_G_loss()
#         self.manual_backward(loss_G)
#         opt_G.step()
#         if opt_F: opt_F.step()
#
#         # Logging
#         self.log_dict({
#             'loss_D': loss_D,
#             'loss_E': loss_E,
#             'loss_G': loss_G
#         }, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
#
#         return loss_G
#
#     def validation_step(self, batch, batch_idx):
#         with torch.no_grad():
#             self.UNSB.set_input(batch)
#             self.UNSB.forward()
#             # 返回或记录 fake_B 做验证
#             return self.UNSB.fake_B
#
#     def configure_optimizers(self):
#         optimizers = []
#         if self.optimizer_G: optimizers.append(self.optimizer_G)
#         if self.optimizer_D: optimizers.append(self.optimizer_D)
#         if self.optimizer_E: optimizers.append(self.optimizer_E)
#         if self.optimizer_F: optimizers.append(self.optimizer_F)
#         return optimizers



class M2PModel(nn.module):
    """
    Lightning wrapper for M2PModel using UNSB (SBModel) and optional BBDM.
    Supports manual optimization for multiple optimizers.
    """
    def __init__(self, options):
        super().__init__()
        self.options = options
        # self.use_modality = options.data.use_modality

        self.encoder1 = VSSMEncoder(embed_dim=64, patch_size=8, d_state=16, expand=2.,
                      compress_ratio=8, squeeze_factor=8, mamba_from_trion=0).cuda()

        # 核心模型
        self.UNSB = SBModel(options.UNSB)

        # 备用模型，可选
        self.BBDM = None
        if use_BBDM and hasattr(options, 'BBDM'):
            self.BBDM = options.BBDM  # 假设 BBDM 有 forward_muti 和 sample 方法

        self.l_train = l_train
        self.l_val = l_val
        self.cuda_num = len(options.train.cuda_num)

        # 手动优化
        self.automatic_optimization = False

        # 提取 UNSB 内部优化器
        self.optimizer_G = getattr(self.UNSB, 'optimizer_G', None)
        self.optimizer_D = getattr(self.UNSB, 'optimizer_D', None)
        self.optimizer_E = getattr(self.UNSB, 'optimizer_E', None)
        self.optimizer_F = getattr(self.UNSB, 'optimizer_F', None)

    def forward(self, input_A, input_B=None):
        # 设置输入
        batch = {'A': input_A}
        if input_B is not None:
            batch['B'] = input_B
        self.UNSB.set_input(batch)
        self.UNSB.forward()
        return self.UNSB.fake_B

    def training_step(self, batch, batch_idx):
        # 手动优化模式必须为 False（你之前设置过 self.automatic_optimization = False）
        optimizers = self.optimizers()
        n_opts = len(optimizers)

        # 期望至少有 G, D, E 三个优化器
        if n_opts < 3:
            raise RuntimeError(f'Expected at least 3 optimizers (G,D,E), got {n_opts}.')

        # 按照 configure_optimizers 中的顺序解包（若有第4个则为 F）
        opt_G = optimizers[0]
        opt_D = optimizers[1]
        opt_E = optimizers[2]
        opt_F = optimizers[3] if n_opts >= 4 else None

        # --- 前向与准备 ---
        # 将 batch 传入 UNSB；你的 SBModel.set_input 接受 dict
        self.UNSB.set_input(batch)
        self.UNSB.forward()

        logs = {}

        # ------------------- update D -------------------
        if hasattr(self.UNSB, 'netD') and self.UNSB.netD is not None:
            self.UNSB.set_requires_grad(self.UNSB.netD, True)
            opt_D.zero_grad()
            loss_D = self.UNSB.compute_D_loss()
            # Lightning 的手动反向
            self.manual_backward(loss_D)
            opt_D.step()
            logs['loss_D'] = loss_D.detach()
        else:
            logs['loss_D'] = torch.tensor(0., device=self.device)

        # ------------------- update E -------------------
        if hasattr(self.UNSB, 'netE') and self.UNSB.netE is not None:
            self.UNSB.set_requires_grad(self.UNSB.netE, True)
            opt_E.zero_grad()
            loss_E = self.UNSB.compute_E_loss()
            self.manual_backward(loss_E)
            opt_E.step()
            logs['loss_E'] = loss_E.detach()
        else:
            logs['loss_E'] = torch.tensor(0., device=self.device)

        # ------------------- update G (& F if exists) -------------------
        # 禁用 D/E 的 grad，然后更新 G（和可能的 F）
        if hasattr(self.UNSB, 'netD'):
            self.UNSB.set_requires_grad(self.UNSB.netD, False)
        if hasattr(self.UNSB, 'netE'):
            self.UNSB.set_requires_grad(self.UNSB.netE, False)

        opt_G.zero_grad()
        if opt_F is not None:
            opt_F.zero_grad()

        loss_G = self.UNSB.compute_G_loss()
        self.manual_backward(loss_G)
        opt_G.step()
        logs['loss_G'] = loss_G.detach()

        if opt_F is not None:
            opt_F.step()
            logs['loss_F'] = getattr(self.UNSB, 'loss_F', torch.tensor(0., device=self.device))
        else:
            logs['loss_F'] = torch.tensor(0., device=self.device)

        # 记录其它你关心的项（例如 SB、NCE 等）
        logs['loss_SB'] = getattr(self.UNSB, 'loss_SB', torch.tensor(0., device=self.device))
        # 如果有更多 loss_names，按需加入 logs

        # 将 logs 写入 Lightning（按 epoch 聚合）
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # 返回用于 Lightning 追踪的主 loss（这里返回 generator loss）
        return loss_G

    def validation_step(self, batch, batch_idx):
        # 使用 UNSB 生成结果
        with torch.no_grad():
            self.UNSB.set_input(batch)
            self.UNSB.forward()
            self.save_imgs(self.UNSB.fake_B, batch['A'], f'val_{batch_idx}')

    def configure_optimizers(self):
        optimizers = []
        if self.optimizer_G: optimizers.append(self.optimizer_G)
        if self.optimizer_D: optimizers.append(self.optimizer_D)
        if self.optimizer_E: optimizers.append(self.optimizer_E)
        if self.optimizer_F: optimizers.append(self.optimizer_F)
        return optimizers

    def save_imgs(self, img_fake, img_target, num):
        """保存 fake 与真实图像对比图"""
        real = img_target[:, 0:1].data
        fake = img_fake[:, 0:1].data
        real = (real - real.min()) / (real.max() - real.min() + 1e-8)
        fake = (fake - fake.min()) / (fake.max() - fake.min() + 1e-8)
        img = torch.cat((real, fake), -2)
        grid = make_grid((img * 255).clip(0, 255))
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
        im = Image.fromarray(ndarr)
        im.convert('L').save(f"logs/images/{num}.png")

class SaveCheck(pl.Callback):
    def __init__(self, options):
        super().__init__()
        self.save_freq = options.train.save_freq
        self.start = options.train.start
        os.makedirs('./logs/checkpoints', exist_ok=True)
        os.makedirs('./logs/images/', exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'Epoch: {pl_module.current_epoch + 1}')
        # 只切换内部网络模式，不直接调用 pl_module.UNSB.train()
        for name in ['netG', 'netD', 'netE', 'netF']:
            net = getattr(pl_module.UNSB, name, None)
            if net is not None:
                net.train()
        self.pbar = tqdm(total=pl_module.l_train // pl_module.cuda_num, ncols=100)

    def on_train_batch_end(self, *args):
        # ✅ 更新进度条
        self.pbar.update(1)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.pbar.close()
        print('val:')
        # 只切换内部网络模式，不直接调用 pl_module.UNSB.eval()
        for name in ['netG', 'netD', 'netE', 'netF']:
            net = getattr(pl_module.UNSB, name, None)
            if net is not None:
                net.eval()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        '''
        if self.start != 0 and self.start == epoch+1:
            pl_module.save_acc.del_more_val()
        pl_module.save_acc.show('val')
        '''
        if (epoch + 1) % self.save_freq == 0 and epoch != 0:
            trainer.save_checkpoint(f"./logs/checkpoints/epoch_{epoch + 1}.ckpt")
            print(f'Save {epoch + 1} last Trainer!')

        print('\n')
        # pl_module.save_acc.save_results()


# class M2PModel(pl.LightningModule):
#     def __init__(self, options, l_train, l_val):
#         super().__init__()
#         self.options = options
#         self.use_modality = options.data.use_modality
#         # self.BBDM = BrownianBridgeModel(options)
#         self.UNSB = SBModel(options.UNSB)
#
#         self.l_train = l_train
#         self.l_val = l_val
#         self.cuda_num = len(options.train.cuda_num)
#
#     def configure_optimizers(self):
#         # opt = torch.optim.Adam(self.UNSB.parameters(), lr=self.options.train.lr, betas=(0.5, 0.999))
#         # return {'optimizer': opt}
#         optimizers = []
#         if hasattr(self, 'optimizer_G'):
#             optimizers.append(self.optimizer_G)
#         if hasattr(self, 'optimizer_D'):
#             optimizers.append(self.optimizer_D)
#         if hasattr(self, 'optimizer_E'):
#             optimizers.append(self.optimizer_E)
#         if hasattr(self, 'optimizer_F'):
#             optimizers.append(self.optimizer_F)
#         return optimizers
#
#     def training_step(self, batch, batch_idx):
#         loss, loss0, loss1, loss2, loss3 = self.BBDM.forward_muti(batch)
#         self.log_dict({'loss': loss.detach(),
#                        'loss0': loss0, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3},
#                       on_step=False, on_epoch=True, sync_dist=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         # simple validation
#         if batch_idx == 0:
#             with torch.no_grad():
#                 context, x_t_1, x_t_2, x_t_3 = self.BBDM.sample(batch)  # B C W H
#                 self.save_imgs(context, batch['PA'], self.current_epoch)
#                 self.save_imgs(x_t_1, batch['PA'], f'{self.current_epoch}_1')
#                 self.save_imgs(x_t_2, batch['PA'], f'{self.current_epoch}_2')
#                 self.save_imgs(x_t_3, batch['PA'], f'{self.current_epoch}_3')
#
#     def save_imgs(self, img_fake, img_target, num):
#         real = img_target[:, 0:1].data
#         fake = img_fake[:, 0:1].data
#         # real, fake = torch.log(real), torch.log(fake)
#         real = (real - real.min()) / (real.max() - real.min())
#         fake = (fake - fake.min()) / (fake.max() - fake.min())
#         img = torch.cat((real, fake), -2)
#         grid = make_grid((img * 255).clip(0, 255))
#         ndarr = grid.permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
#         im = Image.fromarray(ndarr)
#         im.convert('L').save(f"logs/images/{num}.png")
#



# class SaveCheck(pl.Callback):
#     def __init__(self, options):
#         super().__init__()
#         self.save_freq = options.train.save_freq
#         self.start = options.train.start
#         if not os.path.exists(f'./logs/checkpoints'):
#             os.makedirs(f'./logs/checkpoints')
#         if not os.path.exists(f'./logs/images/'):
#             os.makedirs(f'./logs/images/')
#
#     def on_train_epoch_start(self, trainer, pl_module):
#         print(f'Epoch: {pl_module.current_epoch + 1}')
#         pl_module.BBDM.train()
#         self.pbar = tqdm(total=pl_module.l_train // pl_module.cuda_num, ncols=100)
#
#     def on_train_batch_end(self, *args):
#         self.pbar.update(1)
#
#     @rank_zero_only
#     def on_validation_epoch_start(self, trainer, pl_module):
#         self.pbar.close()
#         # pl_module.save_acc.show('train')
#         print('val:')
#         pl_module.BBDM.eval()
#
#     @rank_zero_only
#     def on_validation_epoch_end(self, trainer, pl_module):
#         epoch = trainer.current_epoch
#         '''
#         if self.start != 0 and self.start == epoch+1:
#             pl_module.save_acc.del_more_val()
#         pl_module.save_acc.show('val')
#         '''
#         if (epoch + 1) % self.save_freq == 0 and epoch != 0:
#             trainer.save_checkpoint(f"./logs/checkpoints/epoch_{epoch + 1}.ckpt")
#             print(f'Save {epoch + 1} last Trainer!')
#
#         print('\n')
#         # pl_module.save_acc.save_results()