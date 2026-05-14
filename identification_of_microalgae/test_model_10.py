import gc
import glob
import os
import re
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


MODEL_PATH = r'E:\pythonProject\Microalgae_Identification_YOLOv11\runs\segment\train3\weights\best.pt'
ACTUAL_WIDTH_UM = 44.3
ACTUAL_HEIGHT_UM = 42.8

ROOT_FOLDERS = [
    r'F:\Microalgae_Photoes\20260504\CH1',
    r'F:\Microalgae_Photoes\20260504\CH2',
    r'F:\Microalgae_Photoes\20260504\CH3',
    r'F:\Microalgae_Photoes\20260504\CH4',
    r'F:\Microalgae_Photoes\20260504\CH5',
    r'F:\Microalgae_Photoes\20260504\CH6',
]


def build_predict_attempts():
    if torch.cuda.is_available():
        return [
            {"device": 0, "imgsz": 1024, "max_det": 500, "half": True},
            {"device": 0, "imgsz": 768, "max_det": 400, "half": True},
            {"device": 0, "imgsz": 640, "max_det": 300, "half": False},
            {"device": "cpu", "imgsz": 640, "max_det": 350, "half": False},
        ]
    return [
        {"device": "cpu", "imgsz": 640, "max_det": 350, "half": False},
    ]


def cv2_img_add_text(img, text, position, text_color=(0, 255, 0), text_size=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, text_color, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def calculate_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def merge_overlapping_masks(results, iou_threshold=0.5, class_agnostic=False, area_threshold=10):
    if results is None or results.masks is None or len(results.masks) == 0:
        print("警告: 没有检测到掩膜，无法进行合并")
        return results

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    valid_masks = []
    valid_classes = []
    valid_confs = []
    for mask, cls, conf in zip(masks, classes, confs):
        if np.sum(mask) >= area_threshold:
            valid_masks.append(mask)
            valid_classes.append(cls)
            valid_confs.append(conf)

    if not valid_masks:
        print("警告: 所有掩膜都过小，无法进行合并")
        return results

    masks = np.asarray(valid_masks)
    classes = np.asarray(valid_classes)
    confs = np.asarray(valid_confs)
    num_masks = len(masks)

    merged_flags = np.zeros(num_masks, dtype=bool)
    merged_groups = []
    sorted_indices = np.argsort(confs)[::-1]

    for i in sorted_indices:
        if merged_flags[i]:
            continue

        group = {
            "indices": [i],
            "mask": masks[i].copy(),
            "classes": [classes[i]],
            "confs": [confs[i]],
        }
        merged_flags[i] = True

        for j in sorted_indices:
            if j == i or merged_flags[j]:
                continue

            if not class_agnostic and not np.isclose(classes[j], classes[i]):
                continue

            iou = calculate_iou(group["mask"], masks[j])
            if iou >= iou_threshold:
                group["indices"].append(j)
                group["mask"] = np.logical_or(group["mask"], masks[j])
                group["classes"].append(classes[j])
                group["confs"].append(confs[j])
                merged_flags[j] = True

        merged_groups.append(group)

    final_masks = []
    final_classes = []
    final_confs = []
    final_boxes = []

    for group in merged_groups:
        class_array = np.asarray(group["classes"])
        conf_array = np.asarray(group["confs"], dtype=np.float32)

        unique_classes = np.unique(class_array)
        class_weights = []
        for cls in unique_classes:
            cls_mask = class_array == cls
            avg_conf = np.mean(conf_array[cls_mask]) if np.any(cls_mask) else 0
            class_weights.append(avg_conf * np.sum(cls_mask))

        merged_class = int(unique_classes[np.argmax(class_weights)]) if unique_classes.size > 0 else 0

        weight_sum = np.sum(conf_array)
        if weight_sum > 0:
            merged_conf = float(np.average(conf_array, weights=conf_array))
        else:
            merged_conf = float(np.mean(conf_array)) if len(conf_array) else 0.0

        mask = group["mask"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1, y1, x2, y2 = x, y, x + w, y + h

        h_mask, w_mask = mask.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_mask - 1, int(x2))
        y2 = min(h_mask - 1, int(y2))

        if (x2 - x1) >= 1 and (y2 - y1) >= 1:
            final_masks.append(mask / 255.0)
            final_classes.append(merged_class)
            final_confs.append(merged_conf)
            final_boxes.append([x1, y1, x2, y2])
        else:
            print(f"警告: 合并后的掩膜尺寸过小，已忽略 (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

    if not final_masks:
        print("警告: 合并后没有有效掩膜，返回原始结果")
        return results

    boxes_data = [
        [x1, y1, x2, y2, conf, cls]
        for (x1, y1, x2, y2), conf, cls in zip(final_boxes, final_confs, final_classes)
    ]

    numpy_masks = np.asarray(final_masks, dtype=np.float32)
    torch_masks = torch.tensor(numpy_masks, dtype=torch.float32)
    torch_boxes = torch.tensor(boxes_data, dtype=torch.float32)

    merged_results = Results(
        orig_img=results.orig_img,
        path=results.path,
        names=results.names,
        boxes=torch_boxes,
        masks=torch_masks,
        probs=results.probs,
        speed=getattr(results, "speed", None),
    )

    return merged_results


def is_cuda_oom(exc):
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("cuda" in msg and "out of memory" in msg)


def predict_with_retry(model, image_path):
    errors = []
    attempts = build_predict_attempts()

    for idx, cfg in enumerate(attempts, start=1):
        try:
            print(
                f"  尝试 {idx}/{len(attempts)}: "
                f"device={cfg['device']}, imgsz={cfg['imgsz']}, max_det={cfg['max_det']}, half={cfg['half']}"
            )
            with torch.inference_mode():
                results = model.predict(
                    source=image_path,
                    device=cfg["device"],
                    imgsz=cfg["imgsz"],
                    max_det=cfg["max_det"],
                    half=cfg["half"],
                    verbose=False,
                )
            return results, cfg, None
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            errors.append(err_text)
            print(f"  尝试失败: {err_text}")
            if is_cuda_oom(exc) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return None, None, " | ".join(errors)


def read_image_bgr(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img

    try:
        pil_img = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def sanitize_sheet_name(name):
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
    for ch in invalid_chars:
        name = name.replace(ch, '_')
    name = name.strip()
    return name[:31] if name else "Sheet1"


def make_unique_sheet_name(base_name, used_names):
    base_name = sanitize_sheet_name(base_name)
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    idx = 1
    while True:
        suffix = f"_{idx}"
        candidate = f"{base_name[:31 - len(suffix)]}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def extract_number(filename):
    basename = os.path.basename(filename)
    match = re.search(r'_H(\d+)', basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def append_summary_row(summary_data, img_name, status, target_count=0, total_area="0.00", error_message=""):
    summary_data.append(
        {
            "图片名称": img_name,
            "处理状态": status,
            "目标数量": target_count,
            "总面积(μm²)": total_area,
            "错误信息": error_message,
        }
    )


def process_folder(folder_path, model):
    root_output = os.path.join(folder_path, "results")
    if os.path.exists(root_output):
        try:
            shutil.rmtree(root_output)
        except Exception as exc:
            print(f"警告: 无法删除旧结果目录 {root_output}: {exc}")
    os.makedirs(root_output, exist_ok=True)

    summary_data = []
    failed_data = []
    used_sheet_names = set()

    pattern = r'CH(\d+).*IMG001x0(\d+)'
    match = re.search(pattern, folder_path.replace("/", "\\"))
    if match:
        excel_path = os.path.join(root_output, f"CH{int(match.group(1))}_CB{int(match.group(2))}.xlsx")
    else:
        excel_path = os.path.join(root_output, "test.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
        images = set()
        for ext in image_extensions:
            images.update(glob.glob(os.path.join(folder_path, f"*{ext}")))
            images.update(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        images = sorted(images, key=extract_number)

        if not images:
            print(f"警告: {folder_path} 中未找到图片")
            no_result_df = pd.DataFrame([{"信息": f"文件夹 {os.path.basename(folder_path)} 中未找到图片"}])
            no_result_df.to_excel(writer, sheet_name="无检测结果", index=False)
            return

        print(f"开始处理文件夹: {folder_path}，共 {len(images)} 张图片")

        for image_path in tqdm(images):
            img_name = os.path.basename(image_path)
            sheet_name = make_unique_sheet_name(img_name, used_sheet_names)

            try:
                original_img = read_image_bgr(image_path)
                if original_img is None:
                    raise ValueError("无法读取图像")

                pred_results, cfg_used, pred_error = predict_with_retry(model, image_path)
                if pred_results is None or len(pred_results) == 0:
                    raise RuntimeError(f"推理失败: {pred_error}")

                detection = pred_results[0]
                if detection.boxes is None or len(detection.boxes) == 0:
                    append_summary_row(summary_data, img_name, "未检出", 0, "0.00", "")
                    continue

                result = detection
                if detection.masks is not None and len(detection.masks) > 0:
                    if len(detection.masks) < 30:
                        result = merge_overlapping_masks(detection, iou_threshold=0.3)
                    else:
                        print(f"识别目标超过30，共有{len(detection.masks)}个掩膜，不执行合并掩膜操作")

                if result.masks is None or len(result.masks) == 0:
                    append_summary_row(summary_data, img_name, "未检出", 0, "0.00", "")
                    continue

                pixel_to_um = {
                    "width": ACTUAL_WIDTH_UM / result.orig_img.shape[1],
                    "height": ACTUAL_HEIGHT_UM / result.orig_img.shape[0],
                }

                target_count = 0
                total_area = 0.0
                target_details = []

                for i, mask in enumerate(result.masks.data):
                    target_count += 1
                    class_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    mask_np = mask.cpu().numpy().astype(bool)
                    # mask 分辨率可能比原图低，缩放到原图分辨率再算面积
                    mask_h, mask_w = mask_np.shape
                    orig_h, orig_w = result.orig_img.shape[:2]
                    pixel_scale = (orig_h / mask_h) * (orig_w / mask_w)
                    pixel_area = np.sum(mask_np) * pixel_scale
                    avg_conversion = (pixel_to_um["width"] + pixel_to_um["height"]) / 2
                    actual_area = pixel_area * (avg_conversion ** 2)
                    total_area += actual_area

                    target_details.append(
                        {
                            "目标编号": i + 1,
                            "类别ID": class_id,
                            "置信度": f"{conf:.2f}",
                            "实际面积(μm²)": f"{actual_area:.2f}",
                        }
                    )

                if target_count == 0:
                    append_summary_row(summary_data, img_name, "未检出", 0, "0.00", "")
                    continue

                unannotated_image = result.plot(boxes=False, masks=True)
                annotated_image = result.plot(boxes=False, masks=True)

                stats_text = [
                    f"微藻总数: {target_count}",
                    f"总面积: {total_area:.2f} μm²",
                ]

                y_offset = 30
                for text in stats_text:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(
                        unannotated_image,
                        (10, y_offset - 25),
                        (10 + text_size[0] + 10, y_offset + 5),
                        (0, 0, 0),
                        -1,
                    )
                    unannotated_image = cv2_img_add_text(
                        unannotated_image,
                        text,
                        (15, y_offset - 20),
                        (255, 255, 255),
                        18,
                    )
                    y_offset += 35

                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy().astype(bool)
                    # mask 分辨率可能比原图低，需要缩放到原图坐标
                    mask_h, mask_w = mask_np.shape
                    orig_h, orig_w = result.orig_img.shape[:2]
                    scale_y = orig_h / mask_h
                    scale_x = orig_w / mask_w
                    y, x = np.where(mask_np)
                    if len(x) > 0 and len(y) > 0:
                        x_center = int(np.mean(x) * scale_x)
                        y_center = int(np.mean(y) * scale_y)
                        annotated_image = cv2_img_add_text(
                            annotated_image,
                            f"ID{i + 1}",
                            (x_center - 20, y_center - 20),
                            (0, 255, 0),
                            16,
                        )

                max_height = max(original_img.shape[0], annotated_image.shape[0], unannotated_image.shape[0])
                original_img_resized = cv2.resize(original_img, (original_img.shape[1], max_height))
                annotated_image_resized = cv2.resize(annotated_image, (annotated_image.shape[1], max_height))
                unannotated_image_resized = cv2.resize(unannotated_image, (unannotated_image.shape[1], max_height))
                combined_img = cv2.hconcat([original_img_resized, annotated_image_resized, unannotated_image_resized])

                output_path = os.path.join(root_output, f"R_{img_name}")
                cv2.imwrite(output_path, combined_img)
                print(f"已保存结果图片: {output_path}")

                details_df = pd.DataFrame(target_details)
                details_df.to_excel(writer, sheet_name=sheet_name, index=False)

                stats_df = pd.DataFrame(
                    [
                        {
                            "图片名称": img_name,
                            "原始尺寸(像素)": f"{original_img.shape[1]}x{original_img.shape[0]}",
                            "目标总数": target_count,
                            "总面积(μm²)": f"{total_area:.2f}",
                        }
                    ]
                )
                stats_df.to_excel(writer, sheet_name=sheet_name, startrow=len(details_df) + 3, index=False)

                append_summary_row(summary_data, img_name, "成功", target_count, f"{total_area:.2f}", "")

            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}"
                print(f"处理图片 {image_path} 时出错: {err_text}")
                failed_data.append({"图片名称": img_name, "错误信息": err_text})
                append_summary_row(summary_data, img_name, "失败", 0, "0.00", err_text)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="汇总统计", index=False)
            print(f"已保存汇总报告: {excel_path}")

        if failed_data:
            failed_df = pd.DataFrame(failed_data)
            failed_df.to_excel(writer, sheet_name="失败记录", index=False)

        if not summary_data:
            no_result_df = pd.DataFrame([{"信息": f"文件夹 {os.path.basename(folder_path)} 中所有图片均未检测到目标"}])
            no_result_df.to_excel(writer, sheet_name="无检测结果", index=False)
            print(f"已保存报告: {excel_path}，但所有图片均未检测到目标")

    print(f"文件夹处理完成: {folder_path}")


def process_root_folder(root_folder, model):
    if not os.path.isdir(root_folder):
        print(f"警告: 根目录不存在，已跳过: {root_folder}")
        return

    for subdir in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subfolder_path):
            try:
                process_folder(subfolder_path, model)
            except Exception as exc:
                print(f"处理文件夹 {subfolder_path} 时出错: {type(exc).__name__}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for root_folder in ROOT_FOLDERS:
        process_root_folder(root_folder, model)

    print("所有文件夹处理完成!")


if __name__ == "__main__":
    main()
