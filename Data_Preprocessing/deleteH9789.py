#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : deleteH9789.py
@Time    : 2025/10/3 1:23
@Author  : Pan Yong
@Email   : panyong0417@gmail.com
@Desc    : 
"""
import os


def delete_specific_tif_files(root_dir):
    """
    删除指定文件夹及其子文件夹中所有命名包含H97、H98、H99的.tif图像

    参数:
        root_dir: 根文件夹路径
    """
    # 要匹配的模式
    patterns = ['H97', 'H98', 'H99']

    # 遍历根文件夹及其子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件是否为tif格式且包含指定模式
            if filename.endswith('.tif') and any(pattern in filename for pattern in patterns):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")


if __name__ == "__main__":
    # 替换为你的文件夹路径
    target_directory = r"F:\Microalgae_Photoes\20251002\Processed"

    # 验证文件夹是否存在
    if not os.path.isdir(target_directory):
        print(f"错误: 文件夹 '{target_directory}' 不存在")
    else:
        print(f"开始处理文件夹: {target_directory}")
        delete_specific_tif_files(target_directory)
        print("处理完成")
