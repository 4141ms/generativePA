# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/10/4 14:41 
"""
class Cfg:
    device = 'cuda' # will be resolved in scripts
    img_size = 256
    channels = 1
    batch_size = 8
    lr = 2e-4
    epochs = 200
    time_steps = 1000
    sample_steps = 200
    latent_dim = 64
    z_h = 32
    codebook_size = 512
    vq_commit = 0.25
    train_vqvae = True
    save_dir = './checkpoints_sb_latent'
    physics_loss_weight = 1.0
    structure_loss_weight = 1.0


cfg = Cfg()