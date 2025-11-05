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



def process_nii_to_mip_jpg(input_dir, output_dir):
    """
    处理目录中的所有.nii.gz文件，生成MIP并保存为JPG

    参数:
        input_dir: 输入目录路径（包含.nii.gz文件）
        output_dir: 输出目录路径（将保存JPG文件）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有.nii.gz文件
    nii_files = glob(os.path.join(input_dir, '*.nii.gz'))

    for file_path in nii_files:
        try:
            # 加载NIfTI文件
            img = nib.load(file_path)
            data = img.get_fdata()

            # 获取基础文件名（不含路径和扩展名）
            base_name = os.path.basename(file_path).replace('.nii.gz', '')

            print(f"Processing: {base_name}")

            # 对3D数据做MIP（如果是4D数据需要额外处理）
            if data.ndim == 3:
                # 三个方向的MIP
                mip_axial = np.mean(data, axis=2)  # Z轴投影
                # mip_coronal = np.max(data, axis=1)  # Y轴投影
                # mip_sagittal = np.max(data, axis=0)  # X轴投影

                # 保存三个视角的JPG
                save_mip_as_jpg(mip_axial, os.path.join(output_dir, f"{base_name}_axial.jpg"))
                # save_mip_as_jpg(mip_coronal, os.path.join(output_dir, f"{base_name}_coronal.jpg"))
                # save_mip_as_jpg(mip_sagittal, os.path.join(output_dir, f"{base_name}_sagittal.jpg"))

            elif data.ndim == 4:
                # 处理4D数据（如动态增强MRA）
                print(f"4D data detected - processing time series...")
                for t in range(data.shape[3]):
                    # 对每个时间点做MIP
                    mip = np.max(data[..., t], axis=2)
                    save_mip_as_jpg(mip, os.path.join(output_dir, f"{base_name}_T{t:03d}.jpg"))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


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

# 使用示例
if __name__ == "__main__":
    input_directory = r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA" # 替换为你的输入目录
    output_directory = "../datasets/human/MRI/MRA_mean"  # 替换为输出目录

    # Error processing E:\sd_data\human\3d_brain\zhangf\IXI_MRA\IXI371-IOP-0970-MRA.nii.gz:

    process_nii_to_mip_jpg(input_directory, output_directory)
    print("MIP processing complete!")

    # view_nii()
