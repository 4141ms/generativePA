# main.py 或你的脚本最上方
import warnings

# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
import os
from network.plmodel import MyPlModel, SaveCheck
from utils.util import dict_as_namespace
from data.m2p_dataset import M2PDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import pytorch_lightning as pl
import yaml
import torch
torch.multiprocessing.set_sharing_strategy('file_system')



with open('./configs/options.yaml', 'r') as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

def init(opt):
    """
    初始化
    """
    os.makedirs(opt.checkpoints + '/encoder/', exist_ok=True) # encoder权重
    os.makedirs(opt.checkpoints + '/decoder/', exist_ok=True) # decoder权重

def train_stage0_autoencoder(opt, encoder, decoder, dataloader, device, optim, epochs=5):
    """
    mouse的 MRI 和 PA的不配对数据集
    L_rec在0.08~0.1之间，不收敛，可能需要进行解耦再判断
    """
    encoder.train(); decoder.train()
    l1 = nn.L1Loss()
    images = None
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            m_mri = data['A'].to(device)  # MRI input
            m_pa = data['B'].to(device)    # PA target (used for supervised AE reconstruction in stage0)
            feats = encoder(m_mri)
            # use encoder features to decode directly to PA (end-to-end mapping)
            pa_pred = decoder(feats)
            # print(pa_pred.shape)
            images = pa_pred.cpu().detach()
            L_rec = l1(pa_pred, m_pa)
            optim.zero_grad()
            L_rec.backward()
            optim.step()
            pbar.set_description(f"AE epoch{ep} L_rec:{L_rec.item():.4f}")
        if ep % 5 == 0:
            save_image(images, f'./outputs/image_grid_{ep}.png', nrow=2)
    # with open('{0}/encoder/encoder.pt'.format(opt.checkpoints), 'wb') as f:
    #     torch.save(encoder.state_dict(), f)
    # with open('{0}/decoder/decoder.pt'.format(opt.checkpoints), 'wb') as f:
    #     torch.save(encoder.state_dict(), f)

    return


def train(options):
    torch.autograd.set_detect_anomaly(True)
    options = dict_as_namespace(options)
    root = options.data.niigz_root

    # load dataset
    train_dataset = M2PDataset('train', options.data.use_slice, root, options.data.use_modality)
    val_dataset = M2PDataset('valid', options.data.use_slice, root, options.data.use_modality)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=options.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=options.train.batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=True,
                                persistent_workers=True)
    l_train = len(train_dataloader)
    l_val = len(val_dataloader)

    # other
    SaveCalls = [SaveCheck(options)]
    logger = CSVLogger('./logs', name='loss', flush_logs_every_n_steps=1000)

    # load model
    model_pl = MyPlModel(options, l_train, l_val)
    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        accelerator='gpu',
        devices=options.train.cuda_num,
        max_epochs=options.train.epochs,
        precision=32,
        callbacks=SaveCalls,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=logger,
        log_every_n_steps=5,
        enable_checkpointing=False,
        # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True),
    )

    if options.train.start != 0:
        trainer.fit(
            model=model_pl,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=f"./logs/checkpoints/epoch_{options.train.start}.ckpt"
        )
    else:
        trainer.fit(
            model=model_pl,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )



if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--dataroot', type=str, default='./datasets/MRI2PA/')  # 注意路径
    # parser.add_argument('--phase', type=str, default='train', help='train, test')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    #
    # arg = parser.parse_args()
    # print(options)
    # init(arg)
    train(options)
    # train(arg)

