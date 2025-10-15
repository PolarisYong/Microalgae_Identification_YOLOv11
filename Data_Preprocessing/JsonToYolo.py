import os
import json
import cv2
import numpy as np


def convert_labelme_to_yolo(json_folder, output_folder=None, class_names=None):
    """
    将LabelMe标注的JSON文件转换为YOLO格式的TXT文件

    参数:
        json_folder: 存放LabelMe JSON文件的文件夹路径
        output_folder: 输出YOLO TXT文件的文件夹路径，默认与JSON文件夹相同
        class_names: 类别名称列表，如["class1", "class2"]，若为None则自动从JSON中提取
    """
    # 设置输出文件夹
    if output_folder is None:
        output_folder = json_folder
    os.makedirs(output_folder, exist_ok=True)

    # 收集所有类别名称（如果未提供）
    if class_names is None:
        class_names = []
        # 先遍历所有JSON文件收集类别
        for file in os.listdir(json_folder):
            if file.lower().endswith('.json'):
                json_path = os.path.join(json_folder, file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape['label']
                    if label not in class_names:
                        class_names.append(label)
        # 按字母顺序排序（可选）
        class_names.sort()
        print(f"自动识别类别: {class_names}")
        # 保存类别列表到classes.txt
        with open(os.path.join(output_folder, 'classes.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(class_names))
        print(f"类别列表已保存到: {os.path.join(output_folder, 'classes.txt')}")

    # 遍历所有JSON文件进行转换
    total_converted = 0
    for file in os.listdir(json_folder):
        if file.lower().endswith('.json'):
            json_path = os.path.join(json_folder, file)
            try:
                # 读取JSON数据
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 获取图像尺寸
                image_height = data.get('imageHeight')
                image_width = data.get('imageWidth')

                # 如果JSON中没有图像尺寸，尝试从图像文件获取
                if not image_height or not image_width:
                    image_path = os.path.join(json_folder, data.get('imagePath', ''))
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            image_height, image_width = img.shape[:2]
                        else:
                            print(f"警告: 无法读取图像文件 {image_path}，跳过该JSON")
                            continue
                    else:
                        print(f"警告: JSON {file} 中未找到图像尺寸信息，且图像文件不存在，跳过")
                        continue

                # 准备TXT内容
                txt_content = []
                for shape in data.get('shapes', []):
                    # 获取类别ID
                    label = shape['label']
                    if label not in class_names:
                        print(f"警告: 类别 {label} 不在类别列表中，已跳过")
                        continue
                    class_id = class_names.index(label)

                    # 获取坐标点
                    points = shape['points']
                    if not points or len(points) < 2:
                        print(f"警告: 标注 {label} 坐标点无效，已跳过")
                        continue

                    # 归一化坐标（转换为0-1范围）
                    normalized_points = []
                    for (x, y) in points:
                        # 确保坐标在有效范围内
                        x = max(0, min(x, image_width))
                        y = max(0, min(y, image_height))
                        # 归一化
                        norm_x = x / image_width
                        norm_y = y / image_height
                        normalized_points.append((norm_x, norm_y))

                    # 对于矩形标注（如果是rectangle类型）
                    if shape.get('shape_type') == 'rectangle':
                        # 取第一个点和第二个点作为对角点
                        (x1, y1), (x2, y2) = normalized_points[:2]
                        # 计算中心点和宽高
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = abs(x2 - x1)
                        h = abs(y2 - y1)
                        txt_content.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    else:
                        # 多边形标注（包括line, polygon等）
                        # 格式: class_id x1 y1 x2 y2 ... xn yn
                        poly_str = f"{class_id}"
                        for (x, y) in normalized_points:
                            poly_str += f" {x:.6f} {y:.6f}"
                        txt_content.append(poly_str)

                # 保存为TXT文件（即使内容为空也要生成文件）
                txt_filename = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(output_folder, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(txt_content))
                total_converted += 1
                # 根据是否有内容显示不同信息
                if txt_content:
                    print(f"已转换: {file} -> {txt_filename}")
                else:
                    print(f"已生成空文件: {file} -> {txt_filename}")

            except Exception as e:
                print(f"处理 {file} 时出错: {str(e)}")

    print(f"\n转换完成，共处理 {total_converted} 个JSON文件")
    print(f"YOLO格式TXT文件保存路径: {output_folder}")


if __name__ == "__main__":
    # 配置参数
    json_folder = r"F:\Microalgae_Photoes\20251002\dataset_labelme\CH6"  # LabelMe JSON文件所在文件夹
    output_folder = r"F:\Microalgae_Photoes\20251002\dataset_labelme\CH6"  # 输出YOLO TXT文件的文件夹

    # 如果你知道具体类别名称，可以在这里指定，例如:
    # class_names = ["microalgae", "cell", "other"]
    class_names = None  # 自动从JSON中提取类别

    # 执行转换
    convert_labelme_to_yolo(
        json_folder=json_folder,
        output_folder=output_folder,
        class_names=class_names
    )