""" author: MMDM-Syn"""
from tqdm import tqdm
import random
import numpy as np
import h5py
import scipy.io as sio
import json
import torch
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import os
import nibabel as nib
import yaml
import torchvision.transforms as transforms


import os
import json
import random
import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import zoom


# ==========================
# 🔧 工具函数
# ==========================

def normalize_data(data):
    """归一化到 [0, 1]，并返回归一化结果与参数"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        data_norm = (data - min_val) / (max_val - min_val)
    else:
        data_norm = data
    return data_norm, {'min_val': float(min_val), 'max_val': float(max_val)}


def denormalize_data(data_norm, min_val, max_val):
    """反归一化回原始范围"""
    return data_norm * (max_val - min_val) + min_val


# ==========================
# 🔹 NIfTI 文件读取
# ==========================

def read_data(root, name, modal, slice_idx=None):
    """
    读取指定模态的 .nii.gz 数据，可选择读取单个切片。
    支持懒加载 + 自动归一化 + 参数返回
    """
    nii_path = os.path.join(root, f"IXI_{modal}", f"{name}-{modal}.nii.gz")
    img = nib.load(nii_path, mmap=True)

    try:
        # 懒加载指定切片
        if slice_idx is not None:
            if modal == "T1":
                data_orl = img.dataobj[:, slice_idx, :].astype(np.float32)
                data = np.rot90(data_orl, k=-1)
                data = zoom(data, (1.25, 1), order=1)

                new_shape = (256, 256)
                padded_img = np.zeros(new_shape, dtype=data.dtype)

                # 计算每个维度的起始索引，实现居中填充
                start_h = (new_shape[0] - data.shape[0]) // 2
                start_w = (new_shape[1] - data.shape[1]) // 2

                # 填充
                padded_img[
                    start_h:start_h + data.shape[0],
                    start_w:start_w + data.shape[1]
                ] = data

                data = padded_img
            else:
                data = img.dataobj[:, :, slice_idx].astype(np.float32)
        else:
            # 加载整个体积
            data = img.get_fdata(dtype=np.float32)

        # 归一化到 [0, 1]
        data_norm, params = normalize_data(data)
        data_tran = np.rot90(data_norm, k=-1)

    except Exception as e:
        print(f"⚠️ 文件读取失败: {nii_path}, 错误: {e}")
        data_tran = np.zeros((256, 256), dtype=np.float32)
        params = {'min_val': 0.0, 'max_val': 1.0}

    data_tran = np.fliplr(data_tran)
    return data_tran, params


# ==========================
# 🔹 MAT 文件读取
# ==========================

def read_mat_data(name, slice_idx=None, key='vol'):
    """
    读取 .mat 文件中的体数据（支持懒加载单切片）。
    自动归一化到 [0, 1] 并返回参数。
    """
    root = r'E:\sd_data\results\gt\p0_sos_den'
    if key == 'pressure':
        path = os.path.join(root, "p0", f"{name}-IP.mat")
    elif key == 'sos_map':
        path = os.path.join(root, "sos", f"{name}-SOS.mat")
    elif key == 'den_map':
        path = os.path.join(root, "den", f"{name}-DEN.mat")
    else:
        raise FileNotFoundError(f"未知 key: {key}")

    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"Mat 文件中未找到字段 '{key}'，可用字段为: {list(f.keys())}")

        if slice_idx is not None:
            data_orl = np.transpose(f[key], (1, 2, 0))
            data = data_orl[:, :, slice_idx].astype(np.float32)
        else:
            data = f[key][()].astype(np.float32)

    data_norm, params = normalize_data(data)
    return data_norm, params


# ==========================
# 🧠 数据集类
# ==========================

class M2PDataset(Dataset):
    """
    基于懒加载和归一化参数保存的多模态医学数据集
    """

    def __init__(self, train_val, num_slice, mri_root, use_modality):
        self.mri_root = mri_root
        self.use_modality = use_modality
        self.num_slice = num_slice
        self.t_v = train_val

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])

        # 样本列表
        with open("../87sample.json", 'r') as load_f:
            load_dict = json.load(load_f)
        self.names = load_dict[train_val]

        # 仅记录深度信息
        self.max_len = {name: 50 for name in self.names}

        # 构造训练样本索引
        self.use_data = []
        for name in self.names:
            for core in range(num_slice // 2, self.max_len[name] - num_slice, num_slice // 2):
                self.use_data.append([name, core])
        # random.shuffle(self.use_data)

    def __getitem__(self, idx):
        name, core = self.use_data[idx]
        num = self.num_slice // 2
        batch = {}

        for modal in self.use_modality:
            slice_stack = []
            norm_params = []

            for s in range(core - num, core + num + 1):
                if modal == 'p0':
                    slice_img, params = read_mat_data(name, slice_idx=s, key='pressure')
                elif modal == 'sos':
                    slice_img, params = read_mat_data(name, slice_idx=s, key='sos_map')
                elif modal == 'den':
                    slice_img, params = read_mat_data(name, slice_idx=s, key='den_map')
                elif modal == 'T1':
                    slice_img, params = read_data(self.mri_root, name, modal, slice_idx=s + 128)
                elif modal == 'T2':
                    slice_img, params = read_data(self.mri_root, name, modal, slice_idx=  49 + s )
                elif modal == 'MRA':
                    slice_img, params = read_data(self.mri_root, name, modal, slice_idx= 100 - 49 + s)
                else:
                    raise ValueError(f"未知模态: {modal}")

                slice_stack.append(slice_img)
                norm_params.append(params)

            # 堆叠为 [num_slice, H, W]
            slice_stack = np.stack(slice_stack, axis=-1)
            batch[modal] = self.transform(slice_stack)

            # 保存每个切片的归一化参数（方便反归一化）
            batch[f"{modal}_params"] = norm_params

        return batch

    def __len__(self):
        return len(self.use_data)


def dict_as_namespace(d) -> SimpleNamespace:
    """
    Convert a dictionaty to a namespace (i.e., support for the `.` notation)
    """
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_as_namespace(v))
        else:
            setattr(x, k, v)
    return x

if __name__ == '__main__':
    from utils.util import dict_as_namespace

    with open('../configs/options.yaml', 'r', encoding='utf-8') as f:
        options = yaml.safe_load(f)

    opt = dict_as_namespace(options)
    mri_root = opt.data.mri_root
    mat_root = opt.data.mat_root
    train_dataset = M2PDataset('train', opt.data.use_slice, mri_root, opt.data.use_modality)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True,
                                  persistent_workers=True)

    for epoch in range(0, 20 + 1):
        for i, batch in enumerate(train_dataloader):
            print(i)
            print(batch['T1'].shape)

    k = 30  # 取第10个样本
    data = train_dataset[k]

    t1 = data['T1']
    t1_mip = torch.max(t1, dim=0).values
    t2 = data['T2']
    t2_mip = torch.max(t2, dim=0).values
    mra = data['MRA']
    mra_mip = torch.max(mra, dim=0).values
    p0 = data['p0']
    p0_mip = torch.max(p0, dim=0).values
    sos = data['sos']
    sos_mip = torch.max(sos, dim=0).values
    den = data['den']
    den_mip = torch.max(den, dim=0).values

    import matplotlib.pyplot as plt
    # 放入列表，方便循环绘制
    mip_images = [t1_mip, t2_mip, mra_mip, p0_mip, sos_mip, den_mip]
    titles = ['T1', 'T2', 'MRA', 'p0', 'sos', 'den']
    #
    # # 创建画布
    plt.figure(figsize=(18, 3))

    for i, (img, title) in enumerate(zip(mip_images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(img.cpu(), cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

