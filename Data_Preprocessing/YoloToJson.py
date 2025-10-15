import os
import json
import cv2


def polygon_txt_to_labelme_json(txt_path, image_path, output_json_path=None):
    """
    将多边形格式的txt标签文件转换为LabelMe格式的json文件
    TXT格式：每行第一个值为类别ID，后面是成对的归一化坐标(x1,y1,x2,y2,...xn,yn)
    """
    # 读取图像获取尺寸
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    height, width = img.shape[:2]

    # 读取多边形格式的txt文件
    shapes = []
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 分割数据：第一个是类别ID，后面是坐标对
            parts = list(map(float, line.split()))
            if len(parts) < 5:  # 至少需要1个类别ID + 2对坐标(4个值)
                print(f"跳过无效行: {line}")
                continue

            class_id = int(parts[0])
            coords = parts[1:]  # 提取所有坐标值

            # 检查坐标数量是否为偶数
            if len(coords) % 2 != 0:
                print(f"坐标数量错误（应为偶数）: {line}")
                continue

            # 将归一化坐标转换为像素坐标并组成点列表
            points = []
            for i in range(0, len(coords), 2):
                x_norm = coords[i]
                y_norm = coords[i + 1]

                # 转换为像素坐标
                x = round(x_norm * width)
                y = round(y_norm * height)
                points.append([x, y])

            # 添加多边形标注
            shapes.append({
                "label": str(class_id),  # 可替换为实际类别名称
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",  # 明确指定为多边形
                "flags": {}
            })

    # 构建LabelMe格式的JSON数据
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # 不包含图像数据
        "imageHeight": height,
        "imageWidth": width
    }

    # 确定输出路径
    if output_json_path is None:
        output_json_path = os.path.splitext(txt_path)[0] + ".json"

    # 保存为JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    return output_json_path


def batch_convert(folder_path):
    """批量转换文件夹中所有多边形txt文件为LabelMe json"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for file in os.listdir(folder_path):
        if file.lower().endswith('.txt'):
            txt_path = os.path.join(folder_path, file)
            base_name = os.path.splitext(file)[0]

            # 查找对应的图像文件
            image_path = None
            for ext in image_extensions:
                img_candidate = os.path.join(folder_path, base_name + ext)
                if os.path.exists(img_candidate):
                    image_path = img_candidate
                    break

            if image_path:
                try:
                    json_path = polygon_txt_to_labelme_json(txt_path, image_path)
                    print(f"已转换: {txt_path} -> {json_path}")
                except Exception as e:
                    print(f"转换失败 {txt_path}: {str(e)}")
            else:
                print(f"未找到 {txt_path} 对应的图像文件，跳过")


if __name__ == "__main__":
    # 替换为你的文件夹路径
    target_folder = r"F:\Microalgae_Photoes\datasets_250602\labels"
    batch_convert(target_folder)
    print("转换完成")
