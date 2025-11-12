import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
import json


def n4_bias_correction(img):
    # 确保输入类型为 float32
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    img = corrector.Execute(img, mask)
    return img


def register_to_mra(fixed_img, moving_img):
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # ✅ Execute() 返回最终的 transform
    final_transform = registration.Execute(fixed_img, moving_img)

    # ✅ 用这个 transform 进行重采样
    resampled = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )
    return resampled



def resize_and_pad(volume, target_shape=(256, 256, 256)):
    current_shape = volume.shape
    zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
    volume = zoom(volume, zoom_factors, order=1)
    return volume


def zscore_normalize(volume):
    mean, std = volume.mean(), volume.std()
    normed = (volume - mean) / (std + 1e-8)
    return normed, mean, std


def preprocess_subject(subject_id, paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 读取图像
    t1 = sitk.ReadImage(paths["T1"])
    t2 = sitk.ReadImage(paths["T2"])
    mra = sitk.ReadImage(paths["MRA"])

    # 2️⃣ Bias 校正
    t1 = n4_bias_correction(t1)
    t2 = n4_bias_correction(t2)
    mra = n4_bias_correction(mra)

    # 3️⃣ 以 MRA 为固定图像，配准 T1、T2
    t1_reg = register_to_mra(mra, t1)
    t2_reg = register_to_mra(mra, t2)

    # 4️⃣ 转为 numpy
    t1_np = sitk.GetArrayFromImage(t1_reg)
    t2_np = sitk.GetArrayFromImage(t2_reg)
    mra_np = sitk.GetArrayFromImage(mra)

    # 5️⃣ 重采样/填充
    t1_np = resize_and_pad(t1_np)
    t2_np = resize_and_pad(t2_np)
    mra_np = resize_and_pad(mra_np)

    # 6️⃣ 标准化
    t1_norm, t1_mean, t1_std = zscore_normalize(t1_np)
    t2_norm, t2_mean, t2_std = zscore_normalize(t2_np)
    mra_norm, mra_mean, mra_std = zscore_normalize(mra_np)

    # 7️⃣ 保存结果
    np.save(os.path.join(save_dir, f"{subject_id}_T1.npy"), t1_norm)
    np.save(os.path.join(save_dir, f"{subject_id}_T2.npy"), t2_norm)
    np.save(os.path.join(save_dir, f"{subject_id}_MRA.npy"), mra_norm)

    meta = {
        "T1": {"mean": float(t1_mean), "std": float(t1_std)},
        "T2": {"mean": float(t2_mean), "std": float(t2_std)},
        "MRA": {"mean": float(mra_mean), "std": float(mra_std)},
    }
    with open(os.path.join(save_dir, f"{subject_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"[OK] {subject_id} processed.")


def preprocess_ixi_separate(t1_dir, t2_dir, mra_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 提取所有 ID 的交集
    t1_ids = {os.path.basename(f)[:-10] for f in os.listdir(t1_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}
    t2_ids = {os.path.basename(f)[:-10] for f in os.listdir(t2_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}
    mra_ids = {os.path.basename(f)[:-11] for f in os.listdir(mra_dir) if f.endswith('.nii') or f.endswith('.nii.gz')}

    common_ids = sorted(list(t1_ids & t2_ids & mra_ids))
    print(f"共找到 {len(common_ids)} 个匹配的病例。")

    for sid in tqdm(common_ids):
        try:
            paths = {
                "T1": os.path.join(t1_dir, f"{sid}-T1.nii.gz"),
                "T2": os.path.join(t2_dir, f"{sid}-T2.nii.gz"),
                "MRA": os.path.join(mra_dir, f"{sid}-MRA.nii.gz"),
            }
            # preprocess_subject(sid, paths, save_dir)
            regis_nii(sid, paths, save_dir)
        except Exception as e:
            print(f"[Error] {sid}: {e}")
        # break


def regis_nii(sid, paths, save_dir):
    import ants

    # read / write images
    t1_img = ants.image_read(paths["T1"])
    t2_img = ants.image_read(paths["T2"])
    mra_img = ants.image_read(paths["MRA"])

    # 仿射配准
    aff_t1 = ants.registration(fixed=mra_img, moving=t1_img, type_of_transform='Affine')
    aff_t2 = ants.registration(fixed=mra_img, moving=t2_img, type_of_transform='Affine')

    resliced_t1 = ants.resample_image_to_target(aff_t1['warpedmovout'], mra_img, interp_type='linear')
    resliced_t2 = ants.resample_image_to_target(aff_t2['warpedmovout'], mra_img, interp_type='linear')
    save_path_t1 = os.path.join(save_dir,'T1', f"{sid}-T1.nii.gz")
    save_path_t2 = os.path.join(save_dir, 'T2', f"{sid}-T2.nii.gz")

    ants.image_write(resliced_t1, save_path_t1)
    ants.image_write(resliced_t2, save_path_t2)

    # 可视化三方向切片
    # show_three_planes(fixed_img=t1_img, warped_img=aff['warpedmovout'], overlay_alpha=0.5)



if __name__ == "__main__":
    # t1_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T1\IXI002-Guys-0828-T1.nii.gz"
    t1_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T1"
    t2_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_T2"
    mra_path = r"E:\sd_data\human\3d_brain\zhangf\IXI_MRA"

    t1_list = os.listdir(t1_path)
    print(len(t1_list))

    preprocess_ixi_separate(
        t1_dir=t1_path,
        t2_dir=t2_path,
        mra_dir=mra_path,
        save_dir=r"E:\sd_data\results\IXI_Reg"
    )
