# -*- coding: UTF-8 -*-
"""
@Project ：MySynthesisCode 
@File    ：path.py
@IDE     ：PyCharm 
@Author  ：4141
@Date    ：2025/9/25 11:08 
"""
from pathlib import Path

# 获取当前工具脚本（path_utils.py）的绝对路径
_current_file = Path(__file__).resolve()

# 向上追溯到项目根目录（根据实际目录层级调整 `parent` 次数）
# 例如：path_utils.py 在 src/utils/，则根目录是 src 的父目录
PROJECT_ROOT = _current_file.parent.parent # ../../（根据实际层级修改）

# 定义常用子目录（可选，推荐）
DATA_DIR = PROJECT_ROOT / "data"  # 数据目录（绝对路径）
RAW_DATA_DIR = DATA_DIR / "raw"  # 原始数据目录
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # 处理后数据目录
CONFIG_DIR = PROJECT_ROOT / "configs"  # 配置文件目录