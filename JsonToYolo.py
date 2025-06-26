import os
import json
import numpy as np
import cv2
from glob import glob


def convert_labelme_to_yolo(json_path, img_dir, output_dir, class_dict, save_masks=False):
    """
    将LabelMe JSON标注转换为YOLOv8分割格式

    参数:
    - json_path: JSON文件路径
    - img_dir: 图像目录
    - output_dir: 输出目录
    - class_dict: 类别名称到ID的映射
    - save_masks: 是否保存掩码图像(用于某些需要掩码图像的场景)
    """
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取图像尺寸
    img_filename = data['imagePath']
    img_path = os.path.join(img_dir, img_filename)

    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"警告: 图像 {img_path} 不存在，跳过")
        return

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # 创建YOLO格式的标注内容
    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_dict:
            print(f"警告: 未定义的类别 {label}，跳过")
            continue

        class_id = class_dict[label]

        # 仅处理多边形和折线形状(分割任务)
        if shape['shape_type'] in ['polygon', 'line_string']:
            points = np.array(shape['points'], dtype=np.float32)

            # 归一化多边形点坐标
            normalized_points = []
            for x, y in points:
                normalized_x = x / width
                normalized_y = y / height
                normalized_points.extend([normalized_x, normalized_y])

            # 创建YOLOv8分割格式的行: class_id x1 y1 x2 y2 ...
            yolo_line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
            yolo_lines.append(yolo_line)

            # 可选: 生成掩码图像
            if save_masks:
                mask = np.zeros((height, width), dtype=np.uint8)
                # 确保点是封闭的多边形
                if shape['shape_type'] == 'polygon':
                    pts = points.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                else:  # line_string
                    pts = points.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)

                # 保存掩码图像
                mask_filename = f"{os.path.splitext(img_filename)[0]}_{label}.png"
                mask_path = os.path.join(output_dir, "masks", mask_filename)
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                cv2.imwrite(mask_path, mask)

        else:
            print(f"警告: 不支持的标注形状 {shape['shape_type']}，跳过")

    # 保存YOLO格式的标注文件
    if yolo_lines:
        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, "labels", txt_filename)

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        print(f"已转换: {json_path} -> {txt_path}")


def batch_convert(json_dir, img_dir, output_dir, class_dict, save_masks=False):
    """
    批量转换目录中的所有LabelMe JSON文件

    参数:
    - json_dir: JSON文件目录
    - img_dir: 图像目录
    - output_dir: 输出目录
    - class_dict: 类别映射字典
    - save_masks: 是否保存掩码图像
    """
    # 获取所有JSON文件
    json_files = glob(os.path.join(json_dir, '*.json'))

    # 批量转换
    for json_file in json_files:
        convert_labelme_to_yolo(json_file, img_dir, output_dir, class_dict, save_masks)

    print(f"转换完成! 共处理 {len(json_files)} 个文件")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    json_dir = "jsons/"  # LabelMe JSON文件目录
    img_dir = "images/"  # 图像目录
    output_dir = "data/"  # 输出目录

    # 类别映射字典
    class_dict = {
        "microalgae": 0,
    }

    # 是否保存掩码图像(大多数情况下不需要，YOLOv8可以直接使用坐标)
    save_masks = False

    # 执行批量转换
    batch_convert(json_dir, img_dir, output_dir, class_dict, save_masks)