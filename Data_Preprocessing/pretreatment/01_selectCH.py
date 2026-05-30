#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import shutil


def add_missing_suffix(folder_path):
    """给文件夹下所有不带(XXX)后缀的子文件夹添加(001)后缀"""
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            # 检查是否已有(三位数字)后缀
            if not re.match(r'^.*\(\d{3}\)$', item):
                new_name = f"{item}(001)"
                new_path = os.path.join(folder_path, new_name)
                os.rename(item_path, new_path)
                print(f"重命名: {item} → {new_name}")


def get_week_number(week_folder_name):
    """从周文件夹名称中提取周编号"""
    match = re.match(r'^\d{4}-Week-(\d+)$', week_folder_name)
    return int(match.group(1)) if match else None


def get_sorted_week_folders(photoes_dir):
    """获取按周编号排序的周文件夹列表"""
    week_folders = []
    for item in os.listdir(photoes_dir):
        item_path = os.path.join(photoes_dir, item)
        if os.path.isdir(item_path):
            week_num = get_week_number(item)
            if week_num is not None:
                week_folders.append((week_num, item_path))
    # 按周编号排序
    week_folders.sort(key=lambda x: x[0])
    return [path for _, path in week_folders]


def parse_folder_name(folder_name):
    """解析文件夹名称，返回(基础名称, 编号)"""
    match = re.match(r'^(.*)\((\d{3})\)$', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"文件夹名称格式不正确: {folder_name}")


def get_max_existing_number(target_base_dir):
    """获取目标文件夹中已存在的最大编号"""
    if not os.path.exists(target_base_dir):
        return 0
    max_num = 0
    for item in os.listdir(target_base_dir):
        item_path = os.path.join(target_base_dir, item)
        if os.path.isdir(item_path):
            try:
                _, num = parse_folder_name(item)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue  # 忽略不符合格式的文件夹
    return max_num


def main(photoes_dir):
    if not os.path.exists(photoes_dir):
        print(f"错误：文件夹 {photoes_dir} 不存在")
        return

    # 获取排序后的周文件夹
    sorted_week_folders = get_sorted_week_folders(photoes_dir)
    if not sorted_week_folders:
        print("没有找到符合条件的周文件夹")
        return

    print(f"按顺序处理周文件夹：{[os.path.basename(p) for p in sorted_week_folders]}")

    # 逐个处理周文件夹
    for week_folder in sorted_week_folders:
        week_name = os.path.basename(week_folder)
        print(f"\n开始处理周文件夹：{week_name}")

        # 先处理当前周文件夹内的子文件夹，确保都有编号后缀
        add_missing_suffix(week_folder)
        count = 1
        # 处理每个子文件
        for subfolder in os.listdir(week_folder):
            subfolder_path = os.path.join(week_folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue  # 只处理文件夹

            try:
                base_name, current_num = parse_folder_name(subfolder)
            except ValueError as e:
                print(f"跳过不符合格式的文件夹：{subfolder}，原因：{e}")
                continue

            # 目标基础文件夹路径
            target_base = os.path.join(photoes_dir, base_name)
            os.makedirs(target_base, exist_ok=True)

            # 计算新编号
            if count == 1:
                max_existing = get_max_existing_number(target_base)
                count = 2

            new_num = current_num + max_existing
            new_subfolder = f"{base_name}({new_num:03d})"
            new_path = os.path.join(target_base, new_subfolder)

            # 移动并重命名文件夹
            shutil.move(subfolder_path, new_path)
            print(f"移动完成：{subfolder} → {new_subfolder}")

    print("\n所有文件夹处理完成")


if __name__ == "__main__":
    # 请将此处修改为你的Photoes文件夹实际路径
    photoes_directory = r"F:\Microalgae_Photoes\20260520"
    main(photoes_directory)