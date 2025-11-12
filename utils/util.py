"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from types import SimpleNamespace
import h5py

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)

def view_nii_mlp():
    import numpy as np
    import nibabel as nib

    # 加载nii.gz文件
    img = nib.load('../datasets/human/MRI/IXI002-Guys-0828-MRA.nii.gz')
    data = img.get_fdata()

    # 沿不同方向进行最大值投影
    mip_axial = np.max(data, axis=2)  # 轴位投影
    mip_coronal = np.max(data, axis=1)  # 冠状位投影
    mip_sagittal = np.max(data, axis=0)  # 矢状位投影

    # 保存结果
    nib.save(nib.Nifti1Image(mip_axial, img.affine), '../results/mip_axial.nii.gz')
    nib.save(nib.Nifti1Image(mip_coronal, img.affine), '../results/mip_coronal.nii.gz')
    nib.save(nib.Nifti1Image(mip_sagittal, img.affine), '../results/mip_sagittal.nii.gz')

def view_nii():
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import os

    # 加载nii.gz文件 读取不了0970这个文件
    img = nib.load('../IXI371-IOP-0970-MRA.nii.gz')
    data = img.get_fdata()

    print(data.shape)
    # mip_axial = np.max(data, axis=2)  # Z轴投影
    # save_mip_as_jpg(mip_axial, os.path.join("../datasets/human/MRI/MRA", "IXI371-IOP-0970-MRA_axial.jpg"))

    # # 方法1：自动归一化到0-255（适用于大多数MRA数据）
    # data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    # data_normalized = data_normalized.astype(np.uint8)  # 转换为8位无符号整数
    #
    # # 方法2：手动设置窗宽窗位（如MRA常用窗宽500，窗位300）
    # # window_width = 500
    # # window_center = 300
    # # vmin = window_center - window_width // 2
    # # vmax = window_center + window_width // 2
    # # data_normalized = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    #
    # plt.imsave('../results/mip_output.jpg', data_normalized, cmap='gray', format='jpg')
    #
    # print(data_normalized.shape)
    # print(data_normalized.ndim)

def read_mat_data(slice_idx=None, key='pressure'):
    """
    读取 .mat 格式的体数据（支持懒加载单个切片）。

    参数:
        root: 根目录
        name: 文件名（不带扩展名）
        modal: 模态名称，如 'T1', 'T2' 等
        slice_idx: 指定切片索引，仅加载单张切片
        key: mat 文件中对应的数据字段名，默认 'vol'
    """
    path = os.path.join(r"E:\sd_data\results\gt\p0_sos_den\p0\IXI002-Guys-0828-IP.mat")

    # 使用 h5py 读取 .mat 文件
    with h5py.File(path, 'r') as f:
        if key not in f:
            raise KeyError(f"Mat 文件中未找到字段 '{key}'，可用字段为: {list(f.keys())}")

        dataset = f[key]
        print("dataset.shape", dataset.shape)
        if slice_idx is not None:
            # ✅ 懒加载：只读取某个切片
            data = np.transpose(dataset, (1, 2, 0))
            print("transfer.shape", data.shape)
            data = data[:, :, slice_idx].astype(np.float32)
        else:
            # ✅ 一次性读取整个体积
            data = dataset[()].astype(np.float32)
            data = np.transpose(dataset, (1, 2, 0))


    print("shape:", data.shape)
    print("min:", np.min(data))
    print("max:", np.max(data))
    print("mean:", np.mean(data))
    print("std:", np.std(data))
    print("median:", np.median(data))
    print("原始形状：", data.shape)
    print("强度范围：", np.max(data), "->", np.min(data))

    # mip_xy = np.max(data, axis=2)  # 结果 shape = (256, 256)

    # 可视化
    plt.imshow(data, cmap='gray')
    plt.colorbar(label='Max Intensity')
    plt.title(f'slice-{slice_idx}:(XY plane)')
    plt.show()

    # # 归一化（保持一致）
    # max_val = np.max(data)
    # if max_val > 0:
    #     data = (data - 0.5 * max_val) / (0.5 * max_val)

    return data


def process_nii_to_mip_jpg(input_dir):
    """
    处理目录中的所有.nii.gz文件，生成MIP并保存为JPG

    参数:
        input_dir: 输入目录路径（包含.nii.gz文件）
        output_dir: 输出目录路径（将保存JPG文件）
    """
    from scipy.ndimage import zoom

    # 获取所有.nii.gz文件
    nii_files = glob(os.path.join(input_dir, '*.nii.gz'))
    i = 0
    for file_path in nii_files:
        try:
            # 加载NIfTI文件
            img = nib.load(file_path)
            data = img.get_fdata()
            print("原始形状:", data.shape)
            # 原始 Z 轴 = 150，目标 Z 轴 = 256
            zoom_factors = (1, 1, 256 / 150)  # 高度和宽度保持不变，只缩放 Z
            data = zoom(data, zoom_factors, order=1)  # order=1 双线性插值

            print("Z轴插值后形状:", data.shape)
            # data = np.rot90(data, k=1, axes=(0, 1))  # k=-1 表示顺时针
            # data = np.fliplr(data)  # 沿水平方向翻转

            print("旋转后形状:", data.shape)

            # # 目标大小
            # target_h, target_w = 256, 256
            #
            # # 原始大小
            # H, W, Z = data.shape
            #
            # # 计算缩放比例（只改变每层大小，高度和宽度）
            # zoom_factors = (target_h / H, target_w / W, 1)  # 切片数保持不变
            # data = zoom(data, zoom_factors, order=1)  # order=1 双线性插值
            #
            # print("重采样后形状:", data.shape)

            # 获取基础文件名（不含路径和扩展名）
            base_name = os.path.basename(file_path).replace('.nii.gz', '')
            print("强度范围：", np.max(data), "->", np.min(data))

            # 沿z轴截取中间80个切片
            # num_slices = data.shape[2]
            # center = num_slices // 2
            # half = 80 // 2
            # start = max(center - half, 0)
            # end = min(center + half, num_slices)
            # data = data[:, :, start:end]

            print("截取最后的形状：", data.shape)

            mip_xy = np.max(data, axis=1)  # 结果 shape = (256, 256)

            # 可视化
            plt.imshow(mip_xy, cmap='gray')
            plt.colorbar(label='Max Intensity')
            plt.title(f'MIP{base_name} (XY plane),MRA')
            plt.show()
            plt.close()


            # 对3D数据做MIP（如果是4D数据需要额外处理）
            # if data.ndim == 3:
            #     # 三个方向的MIP
            #     mip_axial = np.mean(data, axis=2)  # Z轴投影
            #     # mip_coronal = np.max(data, axis=1)  # Y轴投影
            #     # mip_sagittal = np.max(data, axis=0)  # X轴投影
            #
            #     # 保存三个视角的JPG
            #     save_mip_as_jpg(mip_axial, os.path.join(output_dir, f"{base_name}_axial.jpg"))
            #     # save_mip_as_jpg(mip_coronal, os.path.join(output_dir, f"{base_name}_coronal.jpg"))
            #     # save_mip_as_jpg(mip_sagittal, os.path.join(output_dir, f"{base_name}_sagittal.jpg"))
            #
            # elif data.ndim == 4:
            #     # 处理4D数据（如动态增强MRA）
            #     print(f"4D data detected - processing time series...")
            #     for t in range(data.shape[3]):
            #         # 对每个时间点做MIP
            #         mip = np.max(data[..., t], axis=2)
            #         save_mip_as_jpg(mip, os.path.join(output_dir, f"{base_name}_T{t:03d}.jpg"))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

        break
        # i = i + 1
        # if i == 5:
        #     break


def load_nifti_ras(file_path, target_size=None, crop_slices=None, visualize=True):
    """
    读取 NIfTI 文件，并对齐到 RAS 方向，返回 3D NumPy 数组

    Args:
        file_path (str): NIfTI 文件路径
        target_size (tuple or None): (H, W) 每个切片目标大小，如果需要插值
        crop_slices (int or None): 中间切片数量，如果需要截取
        visualize (bool): 是否显示中间切片 MIP

    Returns:
        data (np.ndarray): 处理后的 3D 数组，shape = (num_slices, H, W)
    """
    # 1. 读取 NIfTI
    img = nib.load(file_path)
    data = img.get_fdata()

    # 2. 对齐到 RAS 方向
    # nibabel 默认读取数据，不考虑方向；使用 as_closest_canonical 统一到 RAS
    img_canonical = nib.as_closest_canonical(img)
    data = img_canonical.get_fdata()

    # 3. 转为 float32
    data = data.astype(np.float32)

    # 4. 截取中间切片（如果指定）
    if crop_slices is not None and crop_slices < data.shape[0]:
        center = data.shape[0] // 2
        half = crop_slices // 2
        data = data[center - half:center + half, :, :]

    # 5. 插值到目标大小 (H, W)
    if target_size is not None:
        H, W = target_size
        zoom_factors = (H / data.shape[0], H / data.shape[1], W / data.shape[2])
        data = zoom(data, zoom_factors, order=1)

    # 6. 可视化 MIP
    if visualize:
        import matplotlib.pyplot as plt
        mip_xy = np.max(data, axis=2)  # 沿切片轴做 MIP
        plt.imshow(mip_xy, cmap='gray')
        plt.title(f'MIP {file_path}')
        plt.colorbar(label='Intensity')
        plt.show()

    return data

def save_mip_as_jpg(mip_data, output_path, window_width=500, window_center=300):
    """
    将MIP数据保存为JPG图像

    参数:
        mip_data: 2D numpy数组
        output_path: 输出路径
        window_width: 窗宽（对比度调整）
        window_center: 窗位（亮度调整）
    """
    # 窗宽窗位调整（医学影像常用）
    vmin = window_center - window_width // 2
    vmax = window_center + window_width // 2

    # 标准化到0-255范围并裁剪
    mip_normalized = np.clip((mip_data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # 保存为JPG
    plt.imsave(output_path, mip_normalized, cmap='gray', format='jpg')
    plt.close()

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

def read_AP_save_json():
    from pathlib import Path
    import os
    import json
    import random

    folder_path = Path(r'E:\sd_data\results\gt\p0_sos_den\p0')
    save_path = '../87sample.json'
    clean_names = [f.name[:-7] for f in folder_path.iterdir() if f.is_file()]

    random.shuffle(clean_names)

    n = len(clean_names)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)

    train_names = clean_names[:train_end]
    val_names = clean_names[train_end:val_end]
    test_names = clean_names[val_end:]

    split_dict = {
        "train": train_names,
        "test": test_names,
        "valid": val_names,
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ 数据划分完成，共 {n} 个样本")
    print(f"训练集: {len(train_names)}  验证集: {len(val_names)}  测试集: {len(test_names)}")
    print(f"JSON 文件已保存到: {save_path}")

def view_data(niigz_path, mat_path, slice_idx=None):
    import nibabel as nib
    import matplotlib.pyplot as plt

    nii = nib.load(niigz_path)
    data = nii.get_fdata()  # numpy array
    mri = data[:, :, slice_idx]
    print("数据形状:", mri.shape)

    key = 'pressure'
    with h5py.File(mat_path, 'r') as f:
        if key not in f:
            raise KeyError(f"Mat 文件中未找到字段 '{key}'，可用字段为: {list(f.keys())}")

        dataset = f[key]
        if slice_idx is not None:
            # ✅ 懒加载：只读取某个切片
            data_orl = np.transpose(dataset, (1, 2, 0))
            ip = data_orl[:, :, slice_idx].astype(np.float32)
        else:
            # ✅ 一次性读取整个体积
            ip = dataset[()].astype(np.float32)

    print("ip.shape",ip.shape)

    mip_images = [mri, ip]
    titles = ['MRI', 'p0']
    #
    # # 创建画布
    plt.figure(figsize=(6, 3))

    for i, (img, title) in enumerate(zip(mip_images, titles)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def read_copy_data():
    """
    读取demolist里的文件名，并将这些文件复制到另外的地方
    """
    import json
    import shutil

    json_path = "../87sample.json"
    src_dir = r"E:\sd_data\results\IXI_Reg"
    # pa_dir = r"E:\sd_data\results\gt\brain_gt_fluid"
    dst_dir = r"E:\sd_data\results\demo_data"

    os.makedirs(dst_dir, exist_ok=True)

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_filenames = []
    for subset in ["train", "test", "valid"]:
        all_filenames.extend(data[subset])

    for name in all_filenames:

        t1_path = os.path.join(src_dir, f"T1", f"{name}-T1.nii.gz")
        dst_t1_path = os.path.join(dst_dir,f"IXI_T1", f"{name}-T1.nii.gz")

        t2_path = os.path.join(src_dir, f"T2", f"{name}-T2.nii.gz")
        dst_t2_path = os.path.join(dst_dir,f"IXI_T2", f"{name}-T2.nii.gz")

        # mra_path = os.path.join(src_dir, f"IXI_MRA", f"{name}-MRA.nii.gz")
        # dst_mra_path = os.path.join(dst_dir, f"IXI_MRA", f"{name}-MRA.nii.gz")

        if os.path.exists(t1_path):
            shutil.copy2(t1_path, dst_t1_path)
            # shutil.copy2(mra_path, dst_mra_path)
            shutil.copy2(t2_path, dst_t2_path)
            print(f"✅ 已复制 {name}")
        else:
            print(f"⚠️ 文件不存在: {t1_path}")

def view_mri():
    import nibabel as nib

    t1_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T1\IXI002-Guys-0828-T1.nii.gz"
    t2_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T2\IXI002-Guys-0828-T2.nii.gz"
    mra_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI002-Guys-0828-MRA.nii.gz"

    img_T1 = nib.load(t1_path)
    img_T2 = nib.load(t2_path)
    img_MRA = nib.load(mra_path)

    print("T1 shape:", img_T1.shape)
    print("T2 shape:", img_T2.shape)
    print("MRA shape:", img_MRA.shape)

    print("T1 affine:\n", img_T1.affine)
    print("T2 affine:\n", img_T2.affine)
    print("MRA affine:\n", img_MRA.affine)

import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage

def normalize_data(data):
    """归一化到 [0, 1] 并返回参数"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        data_norm = (data - min_val) / (max_val - min_val)
    else:
        data_norm = np.zeros_like(data)
    return data_norm, {'min_val': float(min_val), 'max_val': float(max_val)}

def resample_to_shape(data, target_shape=(256, 256, 80)):
    """
    使用三线性插值将3D体积重采样到指定形状
    data: numpy array (DxHxW 或 HxWxD)
    """
    zoom_factors = np.array(target_shape) / np.array(data.shape)
    data_resampled = ndimage.zoom(data, zoom_factors, order=1)  # order=1 → 三线性插值
    return data_resampled

def read_data(root, name, modal, slice_idx=None):
    """
    读取指定模态的 .nii.gz 数据，可选择读取单个切片。
    支持懒加载 + 自动归一化 + 重采样到 (256,256,80)
    """
    nii_path = os.path.join(root, f"IXI_{modal}", f"{name}-{modal}.nii.gz")
    img = nib.load(nii_path, mmap=True)

    try:
        # 读取数据（懒加载或整体加载）
        if slice_idx is not None:
            data = img.dataobj[:, :, slice_idx].astype(np.float32)
        else:
            data = img.get_fdata(dtype=np.float32)

        # 若形状不一致则重采样
        if data.shape != (256, 256, 80):
            data = resample_to_shape(data, (256, 256, 80))

        # 归一化
        data_norm, params = normalize_data(data)

        # 顺时针旋转90度（若需要）
        data_tran = np.rot90(data_norm, k=-1)

    except Exception as e:
        print(f"⚠️ 文件读取失败: {nii_path}, 错误: {e}")
        data_tran = np.zeros((256, 256), dtype=np.float32)
        params = {'min_val': 0.0, 'max_val': 1.0}

    return data_tran, params

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def resample_to_fixed_size(data, target_shape=(256, 256, 80)):
    """
    将3D MRI数据重采样到指定体积大小。
    data: numpy数组，例如 (256,256,150)
    target_shape: 目标形状 (256,256,80)
    """
    factors = [t / s for t, s in zip(target_shape, data.shape)]
    data_resampled = zoom(data, zoom=factors, order=1)  # order=1为线性插值
    return data_resampled

# ---------- 1️⃣ 工具函数 ----------

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def resample_to_target(data, target_shape=(256, 256, 80)):
    zoom_factors = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2],
    )
    return zoom(data, zoom_factors, order=1)

def load_and_align(nii_path, modal, target_shape=(256, 256, 80)):
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    # ---------- 修正轴顺序 ----------
    if modal == 'T1':
        # T1 内部是 (Y, Z, X)，转成 (X, Y, Z)
        data = np.transpose(data, (2, 0, 1))
        data = normalize(data)
        # 新尺寸
        new_shape = (256, 256, 256)
        padded_img = np.zeros(new_shape, dtype=data.dtype)

        # 计算每个维度的起始索引，实现居中填充
        start_d = (new_shape[0] - data.shape[0]) // 2
        start_h = (new_shape[1] - data.shape[1]) // 2
        start_w = (new_shape[2] - data.shape[2]) // 2

        # 填充
        padded_img[start_d:start_d + data.shape[0],
        start_h:start_h + data.shape[1],
        start_w:start_w + data.shape[2]] = data
        print(f"start_w:{start_w}->start_w + data.shape[2]{start_w + data.shape[2]}")

        print(padded_img.shape)
        return padded_img
    elif modal == 'T2':
        # T2 基本是 (X, Y, Z)
        pass
    elif modal == 'MRA':
        # MRA 是 (X, Y, Z)，但尺寸较大
        if data.shape[0] == 512:
            data = zoom(data, (256/512, 256/512, 1), order=1)

    # ---------- 重采样 & 归一化 ----------
    data = resample_to_target(data, target_shape)
    data = normalize(data)
    return data

def reorient_to_axial(data, orientation):
    if orientation == 'sagittal':  # T1
        data = np.transpose(data, (2, 1, 0))
    elif orientation == 'coronal':  # T2
        data = np.transpose(data, (0, 2, 1))
    # axial: 不变
    return np.flip(data, axis=0)  # 若需要上下翻转


def view_npy(save_flag=False):
    t1 = np.load("../IXI_preprocessed_MRA_fixed/IXI002-Guys-0828_T1.npy")
    t2 = np.load("../IXI_preprocessed_MRA_fixed/IXI002-Guys-0828_T2.npy")
    mra = np.load("../IXI_preprocessed_MRA_fixed/IXI002-Guys-0828_MRA.npy")

    # t1_tran = reorient_to_axial(t1, 'sagittal')
    # t2_tran = reorient_to_axial(t2, 'coronal')

    # if save_flag:
    #     output_nii_path = os.path.join("../IXI_output/")
    #     os.makedirs(output_nii_path, exist_ok=True)
    #
    #     fixed_img = nib.load(r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI033-HH-1259-MRA.nii.gz")
    #     affine = fixed_img.affine
    #
    #     nii = nib.Nifti1Image(mra, affine)
    #
    #     save_path = os.path.join(output_nii_path, "IXI033-HH-1259-reg-mra.nii.gz")
    #
    #     nib.save(nii, save_path)
    # else:
        # return t1_tran, t2_tran, mra

    return t1, t2, mra


import ants
import matplotlib.pyplot as plt


def show_three_planes(fixed_img, moving_img=None, warped_img=None, slice_indices=None, overlay_alpha=0.5):
    """
    显示三方向切片（XY/轴向, XZ/矢状, YZ/冠状）

    fixed_img: ANTsImage, 固定图像（T1）
    moving_img: ANTsImage, 可选，原始移动图像（MRA）
    warped_img: ANTsImage, 可选，配准后的移动图像
    slice_indices: dict, 指定每个方向的切片索引 {'XY': z, 'XZ': y, 'YZ': x}
    overlay_alpha: float, 叠加透明度
    """
    # 转成 numpy
    fixed = fixed_img.numpy()
    warped = warped_img.numpy() if warped_img else None

    # 默认中间切片
    if slice_indices is None:
        slice_indices = {
            'XY': fixed.shape[2] // 2,
            'XZ': fixed.shape[1] // 2,
            'YZ': fixed.shape[0] // 2
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY 平面（轴向）
    z = slice_indices['XY']
    axes[0].imshow(fixed[:, :, z], cmap='gray')
    if warped is not None:
        axes[0].imshow(warped[:, :, z], cmap='hot', alpha=overlay_alpha)
    axes[0].set_title(f'XY plane (Z={z})')
    axes[0].axis('off')

    # XZ 平面（矢状）
    y = slice_indices['XZ']
    axes[1].imshow(fixed[:, y, :], cmap='gray')
    if warped is not None:
        axes[1].imshow(warped[:, y, :], cmap='hot', alpha=overlay_alpha)
    axes[1].set_title(f'XZ plane (Y={y})')
    axes[1].axis('off')

    # YZ 平面（冠状）
    x = slice_indices['YZ']
    axes[2].imshow(fixed[x, :, :], cmap='gray')
    if warped is not None:
        axes[2].imshow(warped[x, :, :], cmap='hot', alpha=overlay_alpha)
    axes[2].set_title(f'YZ plane (X={x})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def regis_nii():
    import ants

    # read / write images
    t1_img = ants.image_read(r"E:\sd_data\human\3d_brain\zhangf\IXI_T1\IXI002-Guys-0828-T1.nii.gz")
    mra_img = ants.image_read(r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI002-Guys-0828-MRA.nii.gz")

    # 仿射配准
    aff = ants.registration(fixed=mra_img, moving=t1_img, type_of_transform='Affine')

    resliced = ants.resample_image_to_target(aff['warpedmovout'], mra_img, interp_type='linear')
    ants.image_write(resliced, "../IXI_output/T1_warped_resliced_to_MRA.nii.gz")

    # 可视化三方向切片
    # show_three_planes(fixed_img=t1_img, warped_img=aff['warpedmovout'], overlay_alpha=0.5)

def regis_save_nii():

    return None


# 使用示例
if __name__ == "__main__":
    import ants
    file_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T1\IXI002-Guys-0828-T1.nii.gz"
    mat_path = r"E:\sd_data\results\gt\p0_sos_den\p0\IXI002-Guys-0828-IP.mat"

    # view_mri()

    # root = r"E:\sd_data\human\3d_brain\zhangf\\"
    # name = "IXI002-Guys-0828"
    # modal = "T1"
    #
    # data, params = read_data(root, name, modal)
    # print(data.shape)  # ✅ (256, 256, 80)
    # print(params)
    #
    # import matplotlib.pyplot as plt
    #
    # # 假设 data 是 (256, 256, 80)
    # slice_idx = 40  # 选取第 40 层
    # plt.imshow(data[:, :, slice_idx], cmap='gray')
    # plt.title(f"Slice {slice_idx}")
    # plt.axis('off')
    # plt.show()

    # img = nib.load(file_path)
    # data = img.get_fdata().astype(np.float32)
    # data_resampled = resample_to_fixed_size(data, (256, 256, 80))
    # print("重采样后形状：", data_resampled.shape)
    # slice_idx = 40  # 选取第 40 层
    # plt.imshow(data[:, :, slice_idx], cmap='gray')
    # plt.title(f"Slice {slice_idx}")
    # plt.axis('off')
    # plt.show()

    # ---------- 2️⃣ 载入你的三个模态 ----------
    t1_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T1\IXI002-Guys-0828-T1.nii.gz"
    t2_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T2\IXI002-Guys-0828-T2.nii.gz"
    mra_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI002-Guys-0828-MRA.nii.gz"

    # t1_tran = ants.image_read(r"E:\sd_data\results\IXI_Reg\T1\IXI002-Guys-0828-T1.nii.gz")
    # t1_tran = ants.image_read("../IXI_output/T1_warped_resliced_to_MRA.nii.gz")
    # mra_img = ants.image_read(r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI002-Guys-0828-MRA.nii.gz")

    # read_mat_data(slice_idx=0)
    read_copy_data()
    # 查看图像维度
    # print(mra_img.shape)

    # 选择一个切片索引，比如 z=75（中间切片）
    # z = t1_tran.shape[2] // 2
    z = 10

    # 显示这个切片
    # ants.plot(mra_img[:, :, z], title=f"MRA registered slice z={z}")


    # regis_nii()
    # t1, t2, mra = view_npy(save_flag=False)
    #
    # t1 = load_and_align(t1_path, "T1")
    # t2 = load_and_align(t2_path, "T2")
    # mra = load_and_align(mra_path, "MRA")

    # ---------- 3️⃣ 可视化 ----------
    # slice_idx = 255  # 查看第40层（Z轴方向）
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(t1[slice_idx,:,  : ], cmap='gray')
    # plt.title(f"T1 - slice {slice_idx}")
    # plt.axis('off')
    # #
    # plt.subplot(1, 3, 2)
    # plt.imshow(t2[slice_idx,:,  : ], cmap='gray')
    # plt.title(f"T2 - slice {slice_idx}")
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(mra[slice_idx, :, : ], cmap='gray')
    # plt.title(f"MRA - slice {slice_idx}")
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(6, 6))


    # slice_idx = 50
    # rgb = np.stack([
    #     t1[:, :, slice_idx],  # R 通道
    #     t2[:, :, slice_idx],  # G 通道
    #     mra[:, :, slice_idx],  # B 通道
    # ], axis=-1)
    #
    #
    #
    # plt.imshow(rgb)
    # plt.title(f"Overlay RGB - slice {slice_idx}")
    # plt.axis('off')
    # plt.show()

