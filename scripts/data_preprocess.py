# -*- coding: UTF-8 -*-
"""
@Project ：MySynthesisCode 
@File    ：data_preprocess.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/9/25 10:17 
"""

import argparse
from pathlib import Path
from bids import BIDSLayout

import pandas as pd

from utils.path import PROJECT_ROOT, RAW_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to directory to save files with paths.")

    args = parser.parse_args()
    return args


def create_ids(args):
    data_dir = RAW_DATA_DIR / 'ds002868'
    layout = BIDSLayout(data_dir, derivatives=False)

    # 获取所有T2w的NIfTI文件路径
    files = layout.get(suffix="T2w", extension="nii.gz", acquisition="RARE", return_type="filename")

    # 构建数据字典列表（兼容MONAI）
    data_list = [{"image": f} for f in files]

    data_df = pd.DataFrame(data_list)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data_list = data_df[:10]
    val_data_list = data_df[10:13]
    test_data_list = data_df[13:]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data_list.to_csv(output_dir / "train.tsv", index=False, sep="\t")
    val_data_list.to_csv(output_dir / "validation.tsv", index=False, sep="\t")
    test_data_list.to_csv(output_dir / "test.tsv", index=False, sep="\t")


if __name__ == '__main__':
    arg = parse_args()
    create_ids(arg)