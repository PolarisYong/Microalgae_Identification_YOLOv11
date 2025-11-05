#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re


def get_hour_num(filename):
    # 正则表达式模式：匹配H后面的数字（捕获组1）
    pattern = r'H(\d+)\.tif'

    # 查找匹配
    match = re.search(pattern, filename)

    if match:
        # 提取捕获组1的内容（即H后面的数字）
        hour_str = match.group(1)
        hour = int(hour_str)  # 转换为整数
        print(f"提取到的数字：{hour}")  # 输出：121
        return hour
    else:
        print("未匹配到数字")
        return -1


def process_subfolder(subfolder_path):
    """处理单个子文件夹，删除小时数>=121的.tif图片"""
    # 收集文件夹中所有.tif文件及其对应的小时数
    tif_info = []
    for filename in os.listdir(subfolder_path):
        # 只处理.tif文件（不区分大小写）
        if filename.lower().endswith('.tif'):
            file_path = os.path.join(subfolder_path, filename)
            # 提取文件名中的小时数（去掉扩展名）
            hour = get_hour_num(file_path)
            tif_info.append((hour, file_path))

    # 筛选出需要删除的文件（小时数>=121）
    to_delete = [file_path for hour, file_path in tif_info if hour >= 121]

    # 执行删除操作
    for file_path in to_delete:
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除失败 {file_path}: {str(e)}")


def main(root_dir):
    # 根目录（photoes文件夹）

    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误：根目录 '{root_dir}' 不存在")
        return

    if not os.path.isdir(root_dir):
        print(f"错误：'{root_dir}' 不是一个文件夹")
        return

    # 遍历根目录下的所有子文件夹
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print(f"\n处理子文件夹: {item_path}")
            process_subfolder(item_path)

    print("\n所有子文件夹处理完毕")


if __name__ == "__main__":
    root_dir = r"F:\Microalgae_Photoes\20251104\CH5"
    main(root_dir)