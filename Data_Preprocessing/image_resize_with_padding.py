#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : image_resize_with_padding.py
@Time    : 2025/7/17 22:38
@Author  : Pan Yong
@Email   : panyong0417@gmail.com
@Desc    : 
"""
import os
from PIL import Image


def resize_image(image_path, target_size):
    """
    将图像 resize 并填充白色背景
    :param image_path: 原始图像路径
    :param target_size: 目标尺寸，元组形式，如 (1224, 1024)
    """
    image = Image.open(image_path)
    # 创建一个指定大小的新图片，填充为白色背景
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    # 将原始图片缩放到合适大小，保持宽高比
    image.thumbnail(target_size)
    # 计算粘贴位置，使原始图像居中
    position = ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)
    # 将原始图像粘贴到新图片上
    new_image.paste(image, position)
    return new_image


def batch_process_images(input_folder, target_size):
    """
    批处理文件夹中的所有图片
    :param input_folder: 输入文件夹路径
    :param target_size: 目标尺寸
    """
    # 创建结果文件夹
    result_folder = os.path.join(input_folder, "result")
    os.makedirs(result_folder, exist_ok=True)

    # 支持的图片格式
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        # 只处理支持的图片格式
        if file_ext in supported_formats:
            input_path = os.path.join(input_folder, filename)
            # 确保不是处理结果文件夹中的文件
            if os.path.isfile(input_path) and not input_path.startswith(result_folder):
                try:
                    # 处理图片
                    resized_image = resize_image(input_path, target_size)
                    # 构建输出文件名
                    output_filename = f"resized_image_{filename}"
                    output_path = os.path.join(result_folder, output_filename)
                    # 保存结果
                    resized_image.save(output_path)
                    print(f"已处理: {filename} -> {output_filename}")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # 用户需要修改此路径为实际的图片文件夹路径
    input_folder = r"E:\pythonProject\Microalgae_Identification_YOLOv11\dataset\other_images"  # 替换为你的图片文件夹路径
    target_size = (1224, 1024)

    batch_process_images(input_folder, target_size)
    print("批处理完成！")