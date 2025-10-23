import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from data.unaligned_dataset import UnalignedDataset

from network.VSSM_Encoder import VSSMEncoder
from network.Decoder import Decoder
from network.LatentHead import LatentProjectionHead

from argparse import ArgumentParser
import os


def init(opt):
    """
    初始化
    """
    os.makedirs(opt.checkpoints + '/encoder/', exist_ok=True) # encoder权重
    os.makedirs(opt.checkpoints + '/decoder/', exist_ok=True) # decoder权重

def train_stage0_autoencoder(opt, encoder, decoder, dataloader, device, optim, epochs=5):
    """
    mouse的 MRI 和 PA的不配对数据集
    """
    encoder.train(); decoder.train()
    l1 = nn.L1Loss()
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for data in pbar:
            m_mri = data['A'].to(device)  # MRI input
            m_pa = data['B'].to(device)    # PA target (used for supervised AE reconstruction in stage0)
            feats = encoder(m_mri)
            # use encoder features to decode directly to PA (end-to-end mapping)
            pa_pred = decoder(feats)
            L_rec = l1(pa_pred, m_pa)
            optim.zero_grad()
            L_rec.backward()
            optim.step()
            pbar.set_description(f"AE epoch{ep} L_rec:{L_rec.item():.4f}")
    with open('{0}/encoder/encoder.pt'.format(opt.checkpoints), 'wb') as f:
        torch.save(encoder.state_dict(), f)
    with open('{0}/decoder/decoder.pt'.format(opt.checkpoints), 'wb') as f:
        torch.save(encoder.state_dict(), f)
    return


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    encoder = VSSMEncoder(embed_dim=64).to(device)
    decoder = Decoder(feature_channel=64).to(device)
    latent_head = LatentProjectionHead(in_channels=64, z_dim=128).to(device)
    # c = torch.randn((1, 3, 128, 128)).cuda()
    # out = encoder.forward(c)
    # latent_feas = latent_head.forward(out)

    opt_ae = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    train_stage0_autoencoder(opt, encoder, decoder, dataloader, device, opt_ae, epochs=5)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./datasets/MRI2PA/')  # 注意路径
    parser.add_argument('--phase', type=str, default='train', help='train, test')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')

    arg = parser.parse_args()
    init(arg)
    train(arg)

