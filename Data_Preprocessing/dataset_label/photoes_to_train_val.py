#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : photoes_to_train_val.py
@Time    : 2025/10/2 21:04
@Author  : Pan Yong
@Email   : panyong0417@gmail.com
@Desc    : 
"""
import os
import shutil
import random
from tqdm import tqdm


def split_dataset(classification_folder, train_size=70):
    """
    将classification文件夹中的文件分割为训练集和验证集

    参数:
        classification_folder: 包含所有文件的主文件夹路径
        train_size: 训练集图片数量，默认70张
    """
    # 创建目标文件夹结构
    folders = [
        "train/pictures",
        "train/json",
        "train/txt",
        "val/pictures",
        "val/json",
        "val/txt"
    ]

    for folder in folders:
        folder_path = os.path.join(classification_folder, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"已创建文件夹: {folder_path}")

    # 获取所有.tif图片文件
    tif_files = [f for f in os.listdir(classification_folder)
                 if f.lower().endswith('.tif') and os.path.isfile(os.path.join(classification_folder, f))]

    if not tif_files:
        print(f"错误: 在 {classification_folder} 中未找到任何.tif图片文件")
        return

    # 检查图片数量是否足够
    total_tif = len(tif_files)
    if total_tif < train_size:
        print(f"警告: 图片总数({total_tif})少于指定的训练集数量({train_size})，将使用所有图片作为训练集")
        train_size = total_tif

    print(f"找到 {total_tif} 张.tif图片，将抽取 {train_size} 张作为训练集，剩余 {total_tif - train_size} 张作为验证集")

    # 随机抽取训练集图片
    random.seed(42)  # 设置随机种子，保证结果可复现
    train_tif = random.sample(tif_files, train_size)
    val_tif = [f for f in tif_files if f not in train_tif]

    # 复制训练集文件
    print("\n开始复制训练集文件...")
    for tif_file in tqdm(train_tif):
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(tif_file)[0]

        # 图片文件路径
        src_tif = os.path.join(classification_folder, tif_file)
        dst_tif = os.path.join(classification_folder, "train/pictures", tif_file)

        # JSON文件路径
        json_file = f"{base_name}.json"
        src_json = os.path.join(classification_folder, json_file)
        dst_json = os.path.join(classification_folder, "train/json", json_file)

        # TXT文件路径
        txt_file = f"{base_name}.txt"
        src_txt = os.path.join(classification_folder, txt_file)
        dst_txt = os.path.join(classification_folder, "train/txt", txt_file)

        # 复制图片
        shutil.copy2(src_tif, dst_tif)

        # 复制JSON（如果存在）
        if os.path.exists(src_json):
            shutil.copy2(src_json, dst_json)
        else:
            print(f"警告: 训练集图片 {tif_file} 对应的JSON文件 {json_file} 不存在")

        # 复制TXT（如果存在）
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            print(f"警告: 训练集图片 {tif_file} 对应的TXT文件 {txt_file} 不存在")

    # 复制验证集文件
    print("\n开始复制验证集文件...")
    for tif_file in tqdm(val_tif):
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(tif_file)[0]

        # 图片文件路径
        src_tif = os.path.join(classification_folder, tif_file)
        dst_tif = os.path.join(classification_folder, "val/pictures", tif_file)

        # JSON文件路径
        json_file = f"{base_name}.json"
        src_json = os.path.join(classification_folder, json_file)
        dst_json = os.path.join(classification_folder, "val/json", json_file)

        # TXT文件路径
        txt_file = f"{base_name}.txt"
        src_txt = os.path.join(classification_folder, txt_file)
        dst_txt = os.path.join(classification_folder, "val/txt", txt_file)

        # 复制图片
        shutil.copy2(src_tif, dst_tif)

        # 复制JSON（如果存在）
        if os.path.exists(src_json):
            shutil.copy2(src_json, dst_json)
        else:
            print(f"警告: 验证集图片 {tif_file} 对应的JSON文件 {json_file} 不存在")

        # 复制TXT（如果存在）
        if os.path.exists(src_txt):
            shutil.copy2(src_txt, dst_txt)
        else:
            print(f"警告: 验证集图片 {tif_file} 对应的TXT文件 {txt_file} 不存在")

    print("\n数据集分割完成!")
    print(f"训练集图片: {len(train_tif)} 张")
    print(f"验证集图片: {len(val_tif)} 张")
    print(f"文件结构已创建在: {classification_folder}")


if __name__ == "__main__":
    # 设置包含所有文件的主文件夹路径
    classification_folder = r"F:\Microalgae_Photoes\20251002\dataset_labelme\CH6"  # 可以替换为绝对路径，如 "C:/data/classification"

    # 检查主文件夹是否存在
    if not os.path.exists(classification_folder) or not os.path.isdir(classification_folder):
        print(f"错误: 文件夹 '{classification_folder}' 不存在或不是有效的目录")
    else:
        # 执行数据集分割，指定训练集大小为70张
        split_dataset(classification_folder, train_size=20)
