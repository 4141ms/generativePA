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


def read_data(root, name, modal, slice_idx=None):
    """
    读取指定模态的 .nii.gz 数据，可选择读取单个切片。
    修改点：
        ✅ 增加 slice_idx 参数，用于懒加载（只读所需切片）
        ✅ 不再一次性加载整个体积，防止内存爆掉
    """
    nii_path = os.path.join(root, f"IXI_{modal}", f"{name}-{modal}.nii.gz")
    img = nib.load(nii_path, mmap=True)

    try:
        if slice_idx is not None:
            # ✅ 只读取一个切片（节省内存）
            data = img.dataobj[:, :, slice_idx].astype(np.float32)
        else:
            # 原逻辑：一次性读取整个体积（内存占用大）
            data = img.get_fdata(dtype=np.float32)

        max_val = np.max(data)
        data = (data - 0.5 * max_val) / (0.5 * max_val)
    except Exception as e:
        print(f"⚠️ 文件读取失败: {nii_path}, 错误: {e}")
        if slice_idx is not None:
            data = np.zeros((256, 256), dtype=np.float32)
        else:
            data = np.zeros((256, 256, 120), dtype=np.float32)  # 体积大小可改

    return data

def read_mat_data(name, slice_idx=None, key='vol'):
    """
    读取 .mat 格式的体数据（支持懒加载单个切片）。

    参数:
        root: 根目录
        name: 文件名（不带扩展名）
        modal: 模态名称，如 'T1', 'T2' 等
        slice_idx: 指定切片索引，仅加载单张切片
        key: mat 文件中对应的数据字段名，默认 'vol'
    """
    path = os.path.join(r'E:\sd_data\results\gt\brain_gt_fluid', f"{name}-S_vol.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 使用 h5py 读取 .mat 文件
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"Mat 文件中未找到字段 '{key}'，可用字段为: {list(f.keys())}")

        dataset = f[key]

        if slice_idx is not None:
            # ✅ 懒加载：只读取某个切片
            data = np.transpose(dataset, (1, 2, 0))
            data = dataset[:, :, slice_idx].astype(np.float32)
        else:
            # ✅ 一次性读取整个体积
            data = dataset[()].astype(np.float32)

    # 归一化（保持一致）
    max_val = np.max(data)
    if max_val > 0:
        data = (data - 0.5 * max_val) / (0.5 * max_val)

    return data

class M2PDataset(Dataset):
    """
    author: MMDM-Syn
    """
    def __init__(self, train_val, num_slice, root, use_modality):
        self.root = root
        self.use_modality = use_modality + ['PA']
        self.num_slice = num_slice
        self.t_v = train_val

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])

        with open("./demolist.json", 'r') as load_f:
            load_dict = json.load(load_f)
        self.names = load_dict[train_val]

        # ✅ 修改：不再加载完整数据，只扫描深度信息
        # 原来这里会把所有模态的3D体积都加载进 self.data，导致内存爆炸
        self.max_len = {}
        for name in tqdm(self.names, desc='Scan depth info'):
            nii_path = os.path.join(root, f"IXI_MRA", f"{name}-MRA.nii.gz")
            img = nib.load(nii_path, mmap=True)
            self.max_len[name] = img.shape[-1]  # 只取深度，不保存实际数据

        # ✅ 构建训练样本索引表
        # 每个样本包含中心切片 core 及其周围 num_slice/2 个切片
        self.use_data = []
        for name in self.names:
            for core in range(num_slice // 2, self.max_len[name] - num_slice, num_slice // 2):
                self.use_data.append([name, core])
        random.shuffle(self.use_data)

    def __getitem__(self, idx):
        """
        ✅ 修改：仅在这里动态加载需要的切片
        原来是直接从 self.data[name][modal] 中取，浪费内存
        """
        name, core = self.use_data[idx]
        num = self.num_slice // 2
        batch = {}

        for modal in self.use_modality:
            # ✅ 每次只读当前窗口范围内的切片
            slice_stack = []
            for s in range(core - num, core + num + 1):
                if modal == 'PA':
                    slice_img = read_mat_data(name, slice_idx=s, key='vol')
                else:
                    slice_img = read_data(self.root, name, modal, slice_idx=s)
                slice_stack.append(slice_img)

            # 将多张切片堆叠为 [H, W, num_slice]
            slice_stack = np.stack(slice_stack, axis=-1)
            batch[modal] = self.transform(slice_stack)

        return batch

    def __len__(self):
        return len(self.use_data)


