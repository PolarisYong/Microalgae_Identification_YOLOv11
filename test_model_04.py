from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import glob
import pandas as pd

# 合并了excel输出和图片输出 03
# 对于单个目标检测出多个掩膜，将掩膜合并
# 加载预训练模型
model = YOLO(r'E:\pythonProject\Microalgae_Identification_YOLOv11\runs\segment\train\weights\best.pt')

# 图片实际尺寸（微米），假设所有图片尺寸一致
actual_width = 44.3
actual_height = 42.8


# 定义中文显示函数
def cv2_img_add_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # Windows
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc"  # macOS
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, text_color, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def merge_overlapping_masks(results, iou_threshold=0.5, class_agnostic=False):
    """
    使用IoU合并高度重叠的实例分割掩膜

    参数:
        results: YOLO模型的检测结果对象
        iou_threshold: IoU阈值，用于判断两个掩膜是否应该合并
        class_agnostic: 是否忽略类别差异进行合并

    返回:
        合并后的结果对象
    """
    if results is None or results.masks is None or len(results.masks) == 0:
        print("警告: 没有检测到掩膜，无法进行合并")
        return results

    # 获取原始掩膜、类别和置信度
    masks = results.masks.data.cpu().numpy()  # 转换为numpy数组
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    num_masks = len(masks)

    # 标记哪些掩膜已经被合并
    merged_flags = [False] * num_masks
    merged_groups = []

    # 遍历所有掩膜，寻找需要合并的组
    for i in range(num_masks):
        if merged_flags[i]:
            continue

        # 创建一个新的合并组
        group = {
            'indices': [i],
            'mask': masks[i].copy(),
            'classes': [classes[i]],
            'confs': [confs[i]]
        }

        # 寻找与当前掩膜重叠超过阈值的其他掩膜
        for j in range(i + 1, num_masks):
            if merged_flags[j]:
                continue

            # 检查类别是否匹配（如果不是类别无关模式）
            if not class_agnostic and classes[j] != classes[i]:
                continue

            # 计算IoU
            iou = calculate_iou(masks[i], masks[j])

            # 如果IoU超过阈值，将此掩膜添加到当前组
            if iou >= iou_threshold:
                group['indices'].append(j)
                group['mask'] = np.logical_or(group['mask'], masks[j])
                group['classes'].append(classes[j])
                group['confs'].append(confs[j])
                merged_flags[j] = True

        # 将合并组添加到结果列表
        merged_groups.append(group)

    # 处理合并后的结果
    final_masks = []
    final_classes = []
    final_confs = []
    final_boxes = []

    for group in merged_groups:
        # 确定合并后的类别（多数投票）
        unique_classes, counts = np.unique(group['classes'], return_counts=True)
        merged_class = unique_classes[np.argmax(counts)] if unique_classes.size > 0 else 0

        # 计算合并后的置信度（加权平均）
        merged_conf = np.mean(group['confs']) if group['confs'] else 0.0

        # 计算合并后的边界框
        mask = group['mask'].astype(np.uint8)
        y, x = np.where(mask)

        # 确保有有效的坐标点
        if len(x) > 0 and len(y) > 0:
            x1, y1, x2, y2 = min(x), min(y), max(x), max(y)

            # 增加边界检查，确保边界框有效
            h, w = mask.shape[:2]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2))
            y2 = min(h - 1, int(y2))

            # 确保边界框有合理的尺寸
            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                final_masks.append(mask)
                final_classes.append(merged_class)
                final_confs.append(merged_conf)
                final_boxes.append([x1, y1, x2, y2])
            else:
                print(f"警告: 合并后的掩膜尺寸过小，已忽略 (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

    # 如果没有找到任何合并结果，返回原始结果
    if not final_masks:
        print("警告: 合并后没有有效掩膜，返回原始结果")
        return results

    # 创建新的结果对象
    from ultralytics.engine.results import Results
    from torch import tensor

    # 创建边界框数据（x1, y1, x2, y2, conf, cls）
    boxes_data = [[x1, y1, x2, y2, conf, cls]
                  for (x1, y1, x2, y2), conf, cls
                  in zip(final_boxes, final_confs, final_classes)]

    # 转换掩膜为PyTorch Tensor（正确的类型转换方式）
    numpy_masks = np.array(final_masks, dtype=np.float32)
    torch_masks = tensor(numpy_masks)

    # 创建新的结果对象
    merged_results = Results(
        orig_img=results.orig_img,
        path=results.path,
        names=results.names,
        boxes=tensor(boxes_data),
        masks=torch_masks,  # 直接使用转换后的Tensor
        probs=results.probs
    )

    return merged_results


def calculate_iou(mask1, mask2):
    """计算两个掩膜之间的交并比(IoU)"""
    # 确保两个掩膜具有相同的尺寸
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8),
                           (mask1.shape[1], mask1.shape[0]))
        mask2 = mask2.astype(bool)

    # 计算交集和并集
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # 避免除零错误
    if union == 0:
        return 0

    return intersection / union


def process_folder(folder_path):
    """处理单个文件夹中的所有图片"""
    # 创建统一结果目录
    root_output = os.path.join(folder_path, "results")
    os.makedirs(root_output, exist_ok=True)

    # 初始化汇总统计
    summary_data = []
    has_valid_images = False  # 标记是否有有效图片（即检测到目标的图片）

    # 创建Excel写入对象，用于保存所有图片的数据
    excel_path = os.path.join(root_output, "微藻细胞分析报告.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 获取文件夹中所有图片
        image_extensions = ['.tif', '.jpg', '.jpeg', '.png']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

        if not images:
            print(f"警告: {folder_path} 中未找到图片")
            # 创建一个"无检测结果"工作表
            no_result_df = pd.DataFrame([{
                '信息': f"文件夹 {os.path.basename(folder_path)} 中未找到图片"
            }])
            no_result_df.to_excel(writer, sheet_name='无检测结果', index=False)
            return

        print(f"开始处理文件夹: {folder_path}，共 {len(images)} 张图片")

        for image_path in tqdm(images):
            try:
                # 读取原图
                original_img = cv2.imread(image_path)
                img_name = os.path.basename(image_path)

                # 生成工作表名称，限制长度避免Excel报错
                sheet_name = img_name[:31]  # Excel工作表名称最大长度为31个字符

                # 目标检测
                results = model(image_path, task='segment')
                if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                    print(f"警告: {image_path} 未检测到目标")
                    # 创建一个"无检测结果"工作表，记录未检测到目标的图片
                    no_result_df = pd.DataFrame([{
                        '图片名称': img_name,
                        '信息': "未检测到目标"
                    }])
                    no_result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                merged_target_num = len(results[0].masks)
                print(f"len(results.masks):{merged_target_num}")
                # 合并重叠掩膜
                result = merge_overlapping_masks(results[0], iou_threshold=0.3)

                merged_target_num2 = len(result.masks)
                print(f"len(resulsda):{merged_target_num2}")

                # 如果合并后没有有效掩膜，跳过后续处理
                if result.masks is None or len(result.masks) == 0:
                    print(f"警告: {image_path} 合并后未检测到有效目标")
                    no_result_df = pd.DataFrame([{
                        '图片名称': img_name,
                        '信息': "合并后未检测到有效目标"
                    }])
                    no_result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                # 计算像素到微米的转换因子
                pixel_to_um = {
                    'width': actual_width / result.orig_img.shape[1],
                    'height': actual_height / result.orig_img.shape[0]
                }

                # 初始化统计变量
                target_details = []
                total_area = 0
                target_count = 0

                # 遍历每个检测目标并计算面积
                if result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        target_count += 1

                        # 获取类别ID和置信度
                        class_id = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])

                        # 将掩码转换为numpy数组
                        mask_np = mask.cpu().numpy().astype(bool)

                        # 计算像素面积
                        pixel_area = np.sum(mask_np)

                        # 计算实际面积
                        avg_conversion = (pixel_to_um['width'] + pixel_to_um['height']) / 2
                        actual_area = pixel_area * (avg_conversion ** 2)
                        total_area += actual_area

                        # 保存目标详细信息
                        target_details.append({
                            '目标编号': i + 1,
                            '类别ID': class_id,
                            '置信度': f"{conf:.2f}",
                            '实际面积(μm²)': f"{actual_area:.2f}"
                        })

                        print(f"图片 {img_name} - 目标 {i + 1}:")
                        print(f"  - 类别: {class_id}")
                        print(f"  - 置信度: {conf:.2f}")
                        print(f"  - 实际面积: {actual_area:.2f} 微米²")

                # 如果没有检测到任何目标，继续处理下一张图片
                if target_count == 0:
                    print(f"警告: {image_path} 未检测到有效目标")
                    no_result_df = pd.DataFrame([{
                        '图片名称': img_name,
                        '信息': "未检测到有效目标"
                    }])
                    no_result_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue

                # 标记有有效图片
                has_valid_images = True

                # 可视化检测结果
                annotated_image = result.plot(boxes=False, masks=True)

                # 在图像上叠加面积和编号信息
                if result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        mask_np = mask.cpu().numpy().astype(bool)
                        pixel_area = np.sum(mask_np)
                        avg_conversion = (pixel_to_um['width'] + pixel_to_um['height']) / 2
                        actual_area = pixel_area * (avg_conversion ** 2)

                        # 获取置信度
                        conf = float(result.boxes.conf[i])

                        # 计算掩码中心位置
                        y, x = np.where(mask_np)
                        if len(x) > 0 and len(y) > 0:
                            x_center, y_center = int(np.mean(x)), int(np.mean(y))

                            # 添加目标编号和置信度
                            annotated_image = cv2_img_add_text(
                                annotated_image,
                                f"ID {i + 1}:({actual_area:.2f}μm2)",
                                (x_center - 40, y_center - 30),
                                (255, 0, 0),  # 蓝色
                                16
                            )

                            # 添加实际面积
                            annotated_image = cv2_img_add_text(
                                annotated_image,
                                f"置信度: {conf:.2f}",
                                (x_center, y_center),
                                (0, 255, 0),  # 绿色
                                16
                            )

                # 拼接原图和结果图（左右拼接）
                max_height = max(original_img.shape[0], annotated_image.shape[0])
                original_img = cv2.resize(original_img, (original_img.shape[1], max_height))
                annotated_image = cv2.resize(annotated_image, (annotated_image.shape[1], max_height))
                combined_img = cv2.hconcat([original_img, annotated_image])

                # 保存拼接后的结果图片
                output_path = os.path.join(root_output, f"combined_{img_name}")
                cv2.imwrite(output_path, combined_img)
                print(f"已保存结果图片: {output_path}")

                # 保存图片统计信息
                img_stats = {
                    '图片名称': img_name,
                    '原始尺寸(像素)': f"{original_img.shape[1]}x{original_img.shape[0]}",
                    '目标总数': target_count,
                    '总面积(μm²)': f"{total_area:.2f}"
                }

                # 在Excel中为当前图片创建一个工作表，写入目标详细信息
                details_df = pd.DataFrame(target_details)
                details_df.to_excel(writer, sheet_name=sheet_name, index=False)

                # 在同一个工作表中追加图片统计信息
                stats_df = pd.DataFrame([img_stats])
                stats_df.to_excel(writer, sheet_name=sheet_name, startrow=len(details_df) + 3, index=False)

                # 更新汇总统计
                summary_data.append({
                    '图片名称': img_name,
                    '目标数量': target_count,
                    '总面积(μm²)': f"{total_area:.2f}"
                })

            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")

        # 保存汇总工作表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
            print(f"已保存汇总报告: {excel_path}")
        elif not has_valid_images:
            # 如果没有有效图片，创建一个"无检测结果"工作表
            no_result_df = pd.DataFrame([{
                '信息': f"文件夹 {os.path.basename(folder_path)} 中所有图片均未检测到目标"
            }])
            no_result_df.to_excel(writer, sheet_name='无检测结果', index=False)
            print(f"已保存报告: {excel_path}，但所有图片均未检测到目标")

    print("所有文件夹处理完成!")


# 指定要处理的文件夹路径
folder_path = r'F:\Microalgae Photoes\20251002\Processed\001_L100_20\IMG001x001'
process_folder(folder_path)

# 示例: 处理多个子文件夹（若有需要可取消注释使用）
# root_folder = r'F:\BaiduNetdiskDownload\target\raw_microalgae_photoes'
# for subdir in os.listdir(root_folder):
#     subfolder_path = os.path.join(root_folder, subdir)
#     if os.path.isdir(subfolder_path):
#         process_folder(subfolder_path)