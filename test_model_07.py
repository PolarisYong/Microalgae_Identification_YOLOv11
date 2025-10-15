from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import glob
import pandas as pd
import shutil

# 对超过30个目标不再进行掩膜合并
# 加载预训练模型
model = YOLO(r'E:\pythonProject\Microalgae_Identification_YOLOv11\runs\segment\train8\weights\best.pt')

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


def merge_overlapping_masks(results, iou_threshold=0.5, class_agnostic=False, area_threshold=10):
    """
    使用IoU合并高度重叠的实例分割掩膜
    优化点：
    1. 增加面积过滤，移除过小掩膜
    2. 使用NMS风格的迭代合并策略，提高效率
    3. 改进合并后类别确定方式，考虑置信度加权
    4. 优化边界框计算
    """
    if results is None or results.masks is None or len(results.masks) == 0:
        print("警告: 没有检测到掩膜，无法进行合并")
        return results

    # 获取原始掩膜、类别和置信度
    masks = results.masks.data.cpu().numpy()  # 转换为numpy数组
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    num_masks = len(masks)

    # 1. 过滤过小的掩膜
    valid_masks = []
    valid_classes = []
    valid_confs = []
    for mask, cls, conf in zip(masks, classes, confs):
        mask_area = np.sum(mask)
        if mask_area >= area_threshold:
            valid_masks.append(mask)
            valid_classes.append(cls)
            valid_confs.append(conf)

    if not valid_masks:
        print("警告: 所有掩膜都过小，无法进行合并")
        return results

    masks = np.array(valid_masks)
    classes = np.array(valid_classes)
    confs = np.array(valid_confs)
    num_masks = len(masks)

    # 标记哪些掩膜已经被合并
    merged_flags = np.zeros(num_masks, dtype=bool)
    merged_groups = []

    # 2. 按置信度排序，优先处理高置信度掩膜
    sorted_indices = np.argsort(confs)[::-1]  # 降序排列

    # 遍历所有掩膜，寻找需要合并的组
    for i in sorted_indices:
        if merged_flags[i]:
            continue

        # 创建一个新的合并组
        group = {
            'indices': [i],
            'mask': masks[i].copy(),
            'classes': [classes[i]],
            'confs': [confs[i]]
        }
        merged_flags[i] = True

        # 寻找与当前掩膜重叠超过阈值的其他掩膜
        for j in sorted_indices:
            if j == i or merged_flags[j]:
                continue

            # 检查类别是否匹配（如果不是类别无关模式）
            if not class_agnostic and not np.isclose(classes[j], classes[i]):
                continue

            # 计算IoU
            iou = calculate_iou(group['mask'], masks[j])

            # 如果IoU超过阈值，将此掩膜添加到当前组
            if iou >= iou_threshold:
                group['indices'].append(j)
                group['mask'] = np.logical_or(group['mask'], masks[j])
                group['classes'].append(classes[j])
                group['confs'].append(confs[j])
                merged_flags[j] = True

        merged_groups.append(group)

    # 处理合并后的结果
    final_masks = []
    final_classes = []
    final_confs = []
    final_boxes = []

    for group in merged_groups:
        # 3. 改进：使用置信度加权确定最终类别
        unique_classes = np.unique(group['classes'])
        class_weights = []

        for cls in unique_classes:
            # 计算该类别的平均置信度
            mask = group['classes'] == cls
            avg_conf = np.mean(np.array(group['confs'])[mask])
            class_weights.append(avg_conf * np.sum(mask))  # 权重 = 平均置信度 * 数量

        merged_class = unique_classes[np.argmax(class_weights)] if unique_classes.size > 0 else 0

        # 计算合并后的置信度（加权平均）
        merged_conf = np.average(group['confs'], weights=group['confs'])  # 用置信度自身作为权重

        # 4. 优化边界框计算，使用更精确的掩码轮廓
        mask = group['mask'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1, y1, x2, y2 = x, y, x + w, y + h

        # 增加边界检查，确保边界框有效
        h_mask, w_mask = mask.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_mask - 1, int(x2))
        y2 = min(h_mask - 1, int(y2))

        # 确保边界框有合理的尺寸
        if (x2 - x1) >= 1 and (y2 - y1) >= 1:
            final_masks.append(mask / 255.0)  # 归一化
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

    # 转换掩膜为PyTorch Tensor
    numpy_masks = np.array(final_masks, dtype=np.float32)
    torch_masks = tensor(numpy_masks)

    # 创建新的结果对象
    merged_results = Results(
        orig_img=results.orig_img,
        path=results.path,
        names=results.names,
        boxes=tensor(boxes_data),
        masks=torch_masks,
        probs=results.probs
    )

    return merged_results


def calculate_iou(mask1, mask2):
    """计算两个掩膜之间的交并比(IoU)"""
    # 确保两个掩膜具有相同的尺寸
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8),
                           (mask1.shape[1], mask1.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
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
    # 若results文件夹已存在，先删除
    if os.path.exists(root_output):
        shutil.rmtree(root_output)
    # 重新创建results文件夹
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
                target_count = 0
                total_area = 0
                target_details = []
                annotated_image = None
                unannotated_image = None
                print('目标检测完成')
                # 检查是否有检测结果
                if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    # 合并重叠掩膜，调整参数适应微藻识别
                    if len(results[0].masks) < 30:
                        result = merge_overlapping_masks(results[0], iou_threshold=0.3)
                    else:
                        print(f"识别目标超过200，共有{len(results[0].masks)}个掩膜，不执行合并掩膜操作")
                        result = results[0]


                    # 检查合并后是否有有效掩膜
                    if result.masks is not None and len(result.masks) > 0:
                        # 计算像素到微米的转换因子
                        pixel_to_um = {
                            'width': actual_width / result.orig_img.shape[1],
                            'height': actual_height / result.orig_img.shape[0]
                        }

                        # 遍历每个检测目标并计算面积
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

                        # 生成无标注的识别结果图（仅包含分割掩码）
                        unannotated_image = result.plot(boxes=False, masks=True)

                        # 在无标注图上添加总数和总面积统计信息
                        stats_text = [
                            f"微藻总数: {target_count}",
                            f"总面积: {total_area:.2f} μm²"
                        ]

                        # 添加统计信息（放在图像左上角，带半透明背景）
                        y_offset = 30
                        for text in stats_text:
                            # 绘制半透明背景
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(
                                unannotated_image,
                                (10, y_offset - 25),
                                (10 + text_size[0] + 10, y_offset + 5),
                                (0, 0, 0),
                                -1
                            )
                            # 添加文字
                            unannotated_image = cv2_img_add_text(
                                unannotated_image,
                                text,
                                (15, y_offset - 20),
                                (255, 255, 255),  # 白色文字
                                18
                            )
                            y_offset += 35

                        # 可视化带标注的检测结果
                        annotated_image = result.plot(boxes=False, masks=True)

                        # 在带标注图上叠加面积和编号信息
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

                                # 添加目标编号和面积
                                annotated_image = cv2_img_add_text(
                                    annotated_image,
                                    f"ID{i + 1}",
                                    (x_center - 40, y_center - 30),
                                    (0, 255, 0),  # 蓝色
                                    16
                                )

                                # 添加置信度
                                # annotated_image = cv2_img_add_text(
                                #     annotated_image,
                                #     f"置信度: {conf:.2f}",
                                #     (x_center, y_center),
                                #     (0, 255, 0),  # 绿色
                                #     16
                                # )
                    else:
                        print(f"警告: {image_path} 合并后未检测到有效目标")
                        # 创建无标注图并添加提示文字
                        unannotated_image = original_img.copy()
                        unannotated_image = cv2_img_add_text(
                            unannotated_image,
                            "合并后未检测到有效目标",
                            (50, 50),
                            (0, 0, 255),  # 红色
                            20
                        )
                        # 创建带标注图
                        annotated_image = unannotated_image.copy()
                else:
                    print(f"警告: {image_path} 未检测到目标")
                    # 创建无标注图并添加提示文字
                    unannotated_image = original_img.copy()
                    unannotated_image = cv2_img_add_text(
                        unannotated_image,
                        "未检测到目标",
                        (50, 50),
                        (0, 0, 255),  # 红色
                        20
                    )
                    # 创建带标注图
                    annotated_image = unannotated_image.copy()

                # 生成三张图的拼接图
                max_height = max(original_img.shape[0], annotated_image.shape[0], unannotated_image.shape[0])
                # 调整高度一致
                original_img_resized = cv2.resize(original_img, (original_img.shape[1], max_height))
                annotated_image_resized = cv2.resize(annotated_image, (annotated_image.shape[1], max_height))
                unannotated_image_resized = cv2.resize(unannotated_image, (unannotated_image.shape[1], max_height))
                # 横向拼接
                combined_img = cv2.hconcat([original_img_resized, annotated_image_resized, unannotated_image_resized])

                # 保存拼接后的结果图片
                output_path = os.path.join(root_output, f"combined_{img_name}")
                cv2.imwrite(output_path, combined_img)
                print(f"已保存结果图片: {output_path}")

                # 准备Excel数据
                if target_count > 0:
                    # 标记有有效图片
                    has_valid_images = True

                    # 保存图片统计信息
                    img_stats = {
                        '图片名称': img_name,
                        '原始尺寸(像素)': f"{original_img.shape[1]}x{original_img.shape[0]}",
                        '目标总数': target_count,
                        '总面积(μm²)': f"{total_area:.2f}"
                    }

                    # 在Excel中为当前图片创建工作表
                    details_df = pd.DataFrame(target_details)
                    details_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # 追加图片统计信息
                    stats_df = pd.DataFrame([img_stats])
                    stats_df.to_excel(writer, sheet_name=sheet_name, startrow=len(details_df) + 3, index=False)

                    # 更新汇总统计
                    summary_data.append({
                        '图片名称': img_name,
                        '目标数量': target_count,
                        '总面积(μm²)': f"{total_area:.2f}"
                    })
                else:
                    # 处理未检测到目标的情况
                    no_result_df = pd.DataFrame([{
                        '图片名称': img_name,
                        '原始尺寸(像素)': f"{original_img.shape[1]}x{original_img.shape[0]}",
                        '信息': "未检测到目标"
                    }])
                    no_result_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # 在汇总统计中记录
                    summary_data.append({
                        '图片名称': img_name,
                        '目标数量': 0,
                        '总面积(μm²)': "0.00"
                    })

            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")

        # 保存汇总工作表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
            print(f"已保存汇总报告: {excel_path}")
        elif not has_valid_images:
            no_result_df = pd.DataFrame([{
                '信息': f"文件夹 {os.path.basename(folder_path)} 中所有图片均未检测到目标"
            }])
            no_result_df.to_excel(writer, sheet_name='无检测结果', index=False)
            print(f"已保存报告: {excel_path}，但所有图片均未检测到目标")

    print("所有文件夹处理完成!")


# 指定要处理的文件夹路径
folder_path = r'F:\Microalgae_Photoes\20251002\Processed\test'
process_folder(folder_path)

# 示例: 处理多个子文件夹（若有需要可取消注释使用）
# root_folder = r'F:\Microalgae_Photoes\20251002\Processed\001_L100_20'
# for subdir in os.listdir(root_folder):
#     subfolder_path = os.path.join(root_folder, subdir)
#     if os.path.isdir(subfolder_path):
#         process_folder(subfolder_path)