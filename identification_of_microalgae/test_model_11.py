import gc
import glob
import os
import re
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

#  无掩码，但是会对微藻细胞边界形态进行修复
MODEL_PATH = r"E:\pythonProject\Microalgae_Identification_YOLOv11\runs\segment\train3\weights\best.pt"

# Update this list if needed.
ROOT_FOLDERS = [
    r"F:\Microalgae_Photoes\text_photoes\CH2",
]

ACTUAL_WIDTH_UM = 44.3
ACTUAL_HEIGHT_UM = 42.8

DENSE_MODE = True
ENABLE_TILE_PASS = True
ENABLE_DENSE_PREPROCESS = True
ENABLE_MASK_REFINEMENT = True

TILE_SIZE = 1024
TILE_OVERLAP = 256

MERGE_IOU_THRESHOLD = 0.35
MIN_MASK_AREA = 10
BOUNDARY_DARK_THRESHOLD = 120
BOUNDARY_ERODE_KERNEL = 7
MIN_INSIDE_RATIO = 0.05
CHAMBER_TEMPLATE_NAMES = ("chamber_template.png", "chamber_mask.png")
CHAMBER_MASK_DILATION = 5
MASK_REPAIR_MAX_KERNEL = 7
MASK_REPAIR_MAX_AREA_GAIN_RATIO = 0.18
MASK_REPAIR_MAX_AREA_LOSS_RATIO = 0.08
MASK_REPAIR_MAX_HOLE_AREA_RATIO = 0.12
MASK_REPAIR_MIN_IOU = 0.78

FONT = cv2.FONT_HERSHEY_SIMPLEX
CPU_MODEL = None


@dataclass
class Instance:
    mask: np.ndarray  # bool mask, local to box
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)
    cls: int
    conf: float

    @property
    def area_pixels(self) -> int:
        return int(self.mask.sum())


def build_predict_attempts(mode="full", cpu_only=False):
    if mode == "tile":
        if cpu_only or not torch.cuda.is_available():
            return [
                {"device": "cpu", "imgsz": 640, "max_det": 500, "half": False, "conf": 0.05, "iou": 0.60, "augment": False, "retina_masks": True},
                {"device": "cpu", "imgsz": 512, "max_det": 400, "half": False, "conf": 0.05, "iou": 0.55, "augment": False, "retina_masks": True},
            ]
        return [
            {"device": 0, "imgsz": 1024, "max_det": 700, "half": True, "conf": 0.03, "iou": 0.65, "augment": False, "retina_masks": True},
            {"device": 0, "imgsz": 896, "max_det": 500, "half": True, "conf": 0.03, "iou": 0.65, "augment": False, "retina_masks": True},
            {"device": 0, "imgsz": 768, "max_det": 400, "half": False, "conf": 0.05, "iou": 0.60, "augment": False, "retina_masks": True},
            {"device": "cpu", "imgsz": 640, "max_det": 500, "half": False, "conf": 0.05, "iou": 0.60, "augment": False, "retina_masks": True},
        ]

    if cpu_only or not torch.cuda.is_available():
        return [
            {"device": "cpu", "imgsz": 768, "max_det": 900, "half": False, "conf": 0.03, "iou": 0.65, "augment": False, "retina_masks": True},
            {"device": "cpu", "imgsz": 640, "max_det": 700, "half": False, "conf": 0.05, "iou": 0.60, "augment": False, "retina_masks": True},
        ]

    return [
        {"device": 0, "imgsz": 1280, "max_det": 1200, "half": True, "conf": 0.03, "iou": 0.70, "augment": True, "retina_masks": True},
        {"device": 0, "imgsz": 1024, "max_det": 900, "half": True, "conf": 0.03, "iou": 0.70, "augment": True, "retina_masks": True},
        {"device": 0, "imgsz": 896, "max_det": 700, "half": False, "conf": 0.05, "iou": 0.65, "augment": False, "retina_masks": True},
        {"device": "cpu", "imgsz": 768, "max_det": 900, "half": False, "conf": 0.03, "iou": 0.65, "augment": False, "retina_masks": True},
    ]


def cv2_img_add_text(img, text, position, text_color=(0, 255, 0), text_size=0.6, thickness=1):
    cv2.putText(img, text, position, FONT, text_size, text_color, thickness, cv2.LINE_AA)
    return img


def load_cjk_font(font_size):
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except Exception:
            continue
    return ImageFont.load_default()


def load_symbol_font(font_size):
    font_paths = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\arialuni.ttf",
        r"C:\Windows\Fonts\seguisym.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size, encoding="utf-8")
        except Exception:
            continue
    return ImageFont.load_default()


def is_cuda_oom(exc):
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("cuda" in msg and "out of memory" in msg)


def is_oom_text(text):
    return "out of memory" in str(text).lower()


def get_cpu_model():
    global CPU_MODEL
    if CPU_MODEL is None:
        CPU_MODEL = YOLO(MODEL_PATH)
    return CPU_MODEL


def safe_empty_cuda_cache():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  warning: failed to empty CUDA cache safely: {type(exc).__name__}: {exc}")


def read_image_bgr(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        return img
    try:
        pil_img = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def preprocess_for_dense(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    return sharpened


def detect_black_wall_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, BOUNDARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOUNDARY_ERODE_KERNEL, BOUNDARY_ERODE_KERNEL))
    barrier = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    barrier = cv2.dilate(barrier, kernel, iterations=1)
    return barrier > 0

def build_auto_chamber_mask(image_bgr):
    wall_mask = detect_black_wall_mask(image_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOUNDARY_ERODE_KERNEL, BOUNDARY_ERODE_KERNEL))
    free = np.where(wall_mask, 0, 255).astype(np.uint8)
    flood = free.copy()
    h, w = free.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed_x, seed_y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed_y, seed_x] == 255:
            cv2.floodFill(flood, flood_mask, (seed_x, seed_y), 128)
            flood_mask.fill(0)

    chamber_mask = flood == 255
    chamber_mask = cv2.erode((chamber_mask.astype(np.uint8) * 255), kernel, iterations=1).astype(bool)
    coverage = float(chamber_mask.mean())
    if coverage < 0.02 or coverage > 0.98:
        return None

    return chamber_mask


def load_template_chamber_mask(image_path, reference_shape):
    search_dirs = [
        os.path.dirname(image_path),
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(),
    ]

    tried_paths = []
    for base_dir in search_dirs:
        for name in CHAMBER_TEMPLATE_NAMES:
            mask_path = os.path.join(base_dir, name)
            tried_paths.append(mask_path)
            if not os.path.exists(mask_path):
                continue

            try:
                mask_img = Image.open(mask_path).convert("L")
                mask = np.asarray(mask_img)
                if mask.shape[:2] != reference_shape:
                    mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_NEAREST)
                return mask > 127
            except Exception as exc:
                print(f"  warning: failed to load chamber template {mask_path}: {type(exc).__name__}: {exc}")

    return None


def warp_bool_mask(mask, dx, dy, out_shape):
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (out_shape[1], out_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped > 0


def mask_iou_bool(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def align_template_to_wall(template_mask, wall_mask):
    if template_mask is None or wall_mask is None:
        return template_mask

    if template_mask.shape != wall_mask.shape:
        template_mask = cv2.resize(
            template_mask.astype(np.uint8),
            (wall_mask.shape[1], wall_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    template_edge = cv2.morphologyEx(template_mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel)
    wall_u8 = (wall_mask.astype(np.uint8) * 255)

    template_f = cv2.GaussianBlur(template_edge.astype(np.float32), (0, 0), 2.0)
    wall_f = cv2.GaussianBlur(wall_u8.astype(np.float32), (0, 0), 2.0)

    try:
        shift, response = cv2.phaseCorrelate(template_f, wall_f)
    except Exception:
        return template_mask

    if not np.isfinite(shift[0]) or not np.isfinite(shift[1]) or response < 0.01:
        return template_mask

    candidates = [
        (shift[0], shift[1]),
        (-shift[0], -shift[1]),
        (0.0, 0.0),
    ]

    best_mask = template_mask
    best_score = -1.0
    wall_score_mask = cv2.dilate(wall_u8, kernel, iterations=1) > 0

    for dx, dy in candidates:
        warped = warp_bool_mask(template_mask, dx, dy, wall_mask.shape[:2])
        warped_edge = cv2.morphologyEx(warped.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel) > 0
        score = mask_iou_bool(cv2.dilate(warped_edge.astype(np.uint8) * 255, kernel, iterations=1) > 0, wall_score_mask)
        if score > best_score:
            best_score = score
            best_mask = warped

    return best_mask


def instance_inside_chamber(instance, chamber_mask):
    x1, y1, x2, y2 = instance.box
    if x2 <= x1 or y2 <= y1:
        return False

    chamber_roi = chamber_mask[y1:y2, x1:x2]
    if chamber_roi.shape != instance.mask.shape:
        chamber_roi = resize_mask_to_shape(chamber_roi, instance.mask.shape[:2])

    total_pixels = int(instance.mask.sum())
    if total_pixels <= 0:
        return False

    inside_pixels = int(np.logical_and(instance.mask, chamber_roi).sum())
    inside_ratio = inside_pixels / total_pixels

    ys, xs = np.where(instance.mask)
    if xs.size == 0 or ys.size == 0:
        return False
    cx = x1 + int(np.mean(xs))
    cy = y1 + int(np.mean(ys))
    if cy < 0 or cx < 0 or cy >= chamber_mask.shape[0] or cx >= chamber_mask.shape[1]:
        return False

    return bool(chamber_mask[cy, cx]) and inside_ratio >= MIN_INSIDE_RATIO


def filter_instances_inside_chamber(instances, chamber_mask):
    if not instances or chamber_mask is None:
        return instances

    filtered = [inst for inst in instances if instance_inside_chamber(inst, chamber_mask)]
    removed = len(instances) - len(filtered)
    if removed > 0:
        print(f"  boundary filter removed {removed} outside instance(s)")
    return filtered


def sanitize_sheet_name(name):
    invalid_chars = ["[", "]", ":", "*", "?", "/", "\\"]
    for ch in invalid_chars:
        name = name.replace(ch, "_")
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
    match = re.search(r"_H(\d+)", basename, re.IGNORECASE)
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


def resize_mask_to_shape(mask, shape_hw):
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def ensure_odd(value):
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


def clamp_kernel_size(value, shape_hw, max_kernel=MASK_REPAIR_MAX_KERNEL):
    max_allowed = min(shape_hw[0], shape_hw[1], max_kernel)
    if max_allowed < 1:
        return 1
    if max_allowed % 2 == 0:
        max_allowed -= 1
    if max_allowed < 1:
        return 1
    return min(ensure_odd(value), max_allowed)


def count_connected_components(mask):
    mask_u8 = mask.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return 0, []
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, num_labels)]
    return len(areas), areas


def compute_hole_stats(mask):
    h, w = mask.shape[:2]
    inv_u8 = (~mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_u8, connectivity=8)
    holes = []

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + comp_w) >= w or (y + comp_h) >= h
        if not touches_border:
            holes.append((idx, area))

    return labels, holes


def fill_small_internal_holes(mask, max_hole_area, max_total_fill_area):
    labels, holes = compute_hole_stats(mask)
    if not holes:
        return mask

    repaired = mask.copy()
    filled_area = 0
    for label_idx, area in sorted(holes, key=lambda item: item[1]):
        if area > max_hole_area or filled_area + area > max_total_fill_area:
            continue
        repaired[labels == label_idx] = True
        filled_area += area

    return repaired


def compute_mask_perimeter(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(cnt, True) for cnt in contours))


def compute_mask_quality(mask):
    component_count, _ = count_connected_components(mask)
    _, holes = compute_hole_stats(mask)
    return {
        "area": int(mask.sum()),
        "components": component_count,
        "hole_count": len(holes),
        "hole_area": int(sum(area for _, area in holes)),
        "perimeter": compute_mask_perimeter(mask),
    }


def tighten_instance_box(instance):
    local_bbox = mask_to_bbox(instance.mask)
    if local_bbox is None:
        return None

    lx1, ly1, lx2, ly2 = local_bbox
    gx1, gy1, _, _ = instance.box
    return Instance(
        mask=instance.mask[ly1:ly2, lx1:lx2].copy(),
        box=(gx1 + lx1, gy1 + ly1, gx1 + lx2, gy1 + ly2),
        cls=instance.cls,
        conf=instance.conf,
    )


def should_accept_mask_candidate(original_mask, candidate_mask):
    if candidate_mask.shape != original_mask.shape:
        return False

    original_stats = compute_mask_quality(original_mask)
    candidate_stats = compute_mask_quality(candidate_mask)

    original_area = max(1, original_stats["area"])
    area_gain_ratio = (candidate_stats["area"] - original_stats["area"]) / float(original_area)
    if area_gain_ratio > MASK_REPAIR_MAX_AREA_GAIN_RATIO or area_gain_ratio < -MASK_REPAIR_MAX_AREA_LOSS_RATIO:
        return False

    overlap_iou = mask_iou_bool(original_mask, candidate_mask)
    if overlap_iou < MASK_REPAIR_MIN_IOU:
        return False

    if candidate_stats["components"] > max(1, original_stats["components"]):
        return False

    improved = False
    if candidate_stats["components"] < original_stats["components"]:
        improved = True
    if candidate_stats["hole_area"] < original_stats["hole_area"]:
        improved = True
    if candidate_stats["hole_count"] < original_stats["hole_count"]:
        improved = True
    if candidate_stats["perimeter"] < original_stats["perimeter"] * 0.985:
        improved = True

    return improved


def refine_single_mask(mask):
    refined = mask.astype(bool)
    if refined.sum() < MIN_MASK_AREA:
        return refined

    area = int(refined.sum())
    bbox_h, bbox_w = refined.shape[:2]
    kernel_size = clamp_kernel_size(np.sqrt(4.0 * area / np.pi) * 0.18, (bbox_h, bbox_w))
    if kernel_size < 3:
        return refined

    max_hole_area = max(4, int(area * MASK_REPAIR_MAX_HOLE_AREA_RATIO))
    max_total_fill_area = max_hole_area * 2

    hole_filled = fill_small_internal_holes(refined, max_hole_area=max_hole_area, max_total_fill_area=max_total_fill_area)
    if should_accept_mask_candidate(refined, hole_filled):
        refined = hole_filled

    pad = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    padded = np.pad(refined.astype(np.uint8) * 255, pad_width=pad, mode="constant", constant_values=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    closed = closed[pad:pad + bbox_h, pad:pad + bbox_w]
    if should_accept_mask_candidate(refined, closed):
        refined = closed

    return refined


def refine_instances(instances):
    refined_instances = []
    for inst in instances:
        refined_mask = refine_single_mask(inst.mask)
        if int(refined_mask.sum()) < MIN_MASK_AREA:
            refined_mask = inst.mask

        tightened = tighten_instance_box(
            Instance(
                mask=refined_mask,
                box=inst.box,
                cls=inst.cls,
                conf=inst.conf,
            )
        )
        if tightened is not None and tightened.area_pixels >= MIN_MASK_AREA:
            refined_instances.append(tightened)
        else:
            refined_instances.append(inst)

    return refined_instances


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def union_box(box_a, box_b):
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def expand_mask_to_box(mask, src_box, dst_box):
    dst_w = dst_box[2] - dst_box[0]
    dst_h = dst_box[3] - dst_box[1]
    canvas = np.zeros((dst_h, dst_w), dtype=bool)

    src_w = src_box[2] - src_box[0]
    src_h = src_box[3] - src_box[1]
    if mask.shape[:2] != (src_h, src_w):
        mask = resize_mask_to_shape(mask, (src_h, src_w))

    dx = src_box[0] - dst_box[0]
    dy = src_box[1] - dst_box[1]
    canvas[dy:dy + src_h, dx:dx + src_w] = mask
    return canvas


def mask_iou(inst_a, inst_b):
    ub = union_box(inst_a.box, inst_b.box)
    a_canvas = expand_mask_to_box(inst_a.mask, inst_a.box, ub)
    b_canvas = expand_mask_to_box(inst_b.mask, inst_b.box, ub)

    intersection = np.logical_and(a_canvas, b_canvas).sum()
    union = np.logical_or(a_canvas, b_canvas).sum()
    if union <= 0:
        return 0.0
    return intersection / union


def build_palette():
    return [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 255, 0),
        (0, 128, 255),
    ]


def overlay_instances(image_bgr, instances, show_labels=False, show_boxes=False, alpha=0.45):
    canvas = image_bgr.copy()
    palette = build_palette()

    for idx, inst in enumerate(instances, start=1):
        x1, y1, x2, y2 = inst.box
        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1
        mask = inst.mask
        if mask.shape[:2] != (box_h, box_w):
            mask = resize_mask_to_shape(mask, (box_h, box_w))

        roi = canvas[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        color = palette[idx % len(palette)]
        color_arr = np.array(color, dtype=np.uint8)

        if mask.any():
            roi_mask = mask
            roi[roi_mask] = (roi[roi_mask] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)

        if show_boxes:
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, 1)

        if show_labels:
            ys, xs = np.where(mask)
            if xs.size > 0 and ys.size > 0:
                cx = x1 + int(xs.mean())
                cy = y1 + int(ys.mean())
                label = str(idx)
                text_size, _ = cv2.getTextSize(label, FONT, 0.4, 1)
                tx = max(0, cx - text_size[0] // 2)
                ty = max(text_size[1] + 1, cy)
                cv2.putText(canvas, label, (tx, ty), FONT, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def add_stats_panel(image_bgr, lines):
    canvas = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    cjk_font = load_cjk_font(24)
    sym_font = load_symbol_font(24)
    x = 10
    y = 10
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=cjk_font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(255, 255, 255))
        if "μm²" in line:
            prefix, suffix = line.split("μm²", 1)
            prefix_text = prefix + "μm"
            draw.text((x, y), prefix_text, fill=(0, 0, 0), font=cjk_font)
            prefix_bbox = draw.textbbox((x, y), prefix_text, font=cjk_font)
            draw.text((prefix_bbox[2], y), "²", fill=(0, 0, 0), font=sym_font)
            if suffix:
                sym_bbox = draw.textbbox((prefix_bbox[2], y), "²", font=sym_font)
                draw.text((sym_bbox[2], y), suffix, fill=(0, 0, 0), font=cjk_font)
        else:
            draw.text((x, y), line, fill=(0, 0, 0), font=cjk_font)
        y += (bbox[3] - bbox[1]) + 10
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def iter_tiles(image_shape, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    h, w = image_shape[:2]
    step = max(1, tile_size - overlap)

    x_starts = list(range(0, max(w - tile_size, 0) + 1, step))
    y_starts = list(range(0, max(h - tile_size, 0) + 1, step))

    if not x_starts:
        x_starts = [0]
    if not y_starts:
        y_starts = [0]

    last_x = max(0, w - tile_size)
    last_y = max(0, h - tile_size)
    if x_starts[-1] != last_x:
        x_starts.append(last_x)
    if y_starts[-1] != last_y:
        y_starts.append(last_y)

    x_starts = sorted(set(x_starts))
    y_starts = sorted(set(y_starts))

    for y0 in y_starts:
        y1 = min(h, y0 + tile_size)
        for x0 in x_starts:
            x1 = min(w, x0 + tile_size)
            yield x0, y0, x1, y1


def predict_with_retry(model, source, mode="full", cpu_only=False):
    errors = []
    had_oom = False
    attempts = build_predict_attempts(mode=mode, cpu_only=cpu_only)

    for idx, cfg in enumerate(attempts, start=1):
        try:
            print(
                f"  attempt {idx}/{len(attempts)}: "
                f"device={cfg['device']}, imgsz={cfg['imgsz']}, max_det={cfg['max_det']}, "
                f"conf={cfg['conf']}, iou={cfg['iou']}, half={cfg['half']}, augment={cfg['augment']}"
            )
            active_model = get_cpu_model() if cfg["device"] == "cpu" else model
            with torch.inference_mode():
                results = active_model.predict(
                    source=source,
                    device=cfg["device"],
                    imgsz=cfg["imgsz"],
                    max_det=cfg["max_det"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                    half=cfg["half"],
                    augment=cfg["augment"],
                    retina_masks=cfg["retina_masks"],
                    verbose=False,
                )
            return results, cfg, None, had_oom
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            errors.append(err_text)
            print(f"  failed: {err_text}")
            if is_cuda_oom(exc):
                had_oom = True
            if is_cuda_oom(exc) and torch.cuda.is_available():
                safe_empty_cuda_cache()
            gc.collect()

    return None, None, " | ".join(errors), had_oom


def extract_instances_from_result(result, offset_x=0, offset_y=0):
    instances = []
    if result is None or result.masks is None or len(result.masks) == 0:
        return instances

    source_h, source_w = result.orig_img.shape[:2]
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.zeros(len(masks))
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))

    for mask, cls, conf in zip(masks, classes, confs):
        mask = resize_mask_to_shape(mask, (source_h, source_w))
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        mask_crop = mask[y1:y2, x1:x2].astype(bool)
        if mask_crop.sum() < MIN_MASK_AREA:
            continue

        instances.append(
            Instance(
                mask=mask_crop,
                box=(offset_x + x1, offset_y + y1, offset_x + x2, offset_y + y2),
                cls=int(cls),
                conf=float(conf),
            )
        )

    return instances


def predict_tiles(model, image_bgr, cpu_only=False):
    h, w = image_bgr.shape[:2]
    all_instances = []
    had_oom = False
    tile_size = 512 if cpu_only else TILE_SIZE
    tile_overlap = 128 if cpu_only else TILE_OVERLAP

    for x0, y0, x1, y1 in iter_tiles((h, w), tile_size, tile_overlap):
        tile = image_bgr[y0:y1, x0:x1]
        if tile.size == 0:
            continue

        tile_results, cfg_used, pred_error, tile_had_oom = predict_with_retry(model, tile, mode="tile", cpu_only=cpu_only)
        had_oom = had_oom or tile_had_oom or is_oom_text(pred_error)
        if tile_results is None or len(tile_results) == 0:
            print(f"  tile failed at ({x0}, {y0}) - ({x1}, {y1}): {pred_error}")
            continue

        det = tile_results[0]
        tile_instances = extract_instances_from_result(det, offset_x=x0, offset_y=y0)
        all_instances.extend(tile_instances)

    return all_instances, had_oom


def merge_instances(instances, iou_threshold=MERGE_IOU_THRESHOLD, class_agnostic=True):
    if not instances:
        return []

    instances = [inst for inst in instances if inst.area_pixels >= MIN_MASK_AREA]
    if not instances:
        return []

    sorted_indices = np.argsort([inst.conf for inst in instances])[::-1]
    used = np.zeros(len(instances), dtype=bool)
    merged = []

    for i in sorted_indices:
        if used[i]:
            continue

        group_box = instances[i].box
        group_mask = instances[i].mask.copy()
        group_classes = [instances[i].cls]
        group_confs = [instances[i].conf]
        used[i] = True

        for j in sorted_indices:
            if used[j] or j == i:
                continue

            if (not class_agnostic) and instances[j].cls != instances[i].cls:
                continue

            candidate = instances[j]
            if box_iou(group_box, candidate.box) == 0:
                continue

            union = union_box(group_box, candidate.box)
            group_canvas = expand_mask_to_box(group_mask, group_box, union)
            cand_canvas = expand_mask_to_box(candidate.mask, candidate.box, union)
            iou = mask_iou(
                Instance(mask=group_canvas, box=union, cls=instances[i].cls, conf=instances[i].conf),
                Instance(mask=cand_canvas, box=union, cls=candidate.cls, conf=candidate.conf),
            )
            if iou >= iou_threshold:
                group_box = union
                group_mask = np.logical_or(group_canvas, cand_canvas)
                group_classes.append(candidate.cls)
                group_confs.append(candidate.conf)
                used[j] = True

        classes = np.asarray(group_classes)
        confs = np.asarray(group_confs, dtype=np.float32)
        unique_classes = np.unique(classes)
        if unique_classes.size == 1:
            merged_cls = int(unique_classes[0])
        else:
            class_scores = []
            for cls in unique_classes:
                cls_mask = classes == cls
                class_scores.append(float(np.sum(confs[cls_mask])))
            merged_cls = int(unique_classes[int(np.argmax(class_scores))])

        weight_sum = float(np.sum(confs))
        if weight_sum > 0:
            merged_conf = float(np.average(confs, weights=np.maximum(confs, 1e-6)))
        else:
            merged_conf = float(np.mean(confs)) if len(confs) else 0.0

        merged.append(Instance(mask=group_mask, box=group_box, cls=merged_cls, conf=merged_conf))

    return merged


def process_image(model, image_path):
    original_img = read_image_bgr(image_path)
    if original_img is None:
        raise ValueError("unable to read image")

    working_img = preprocess_for_dense(original_img) if ENABLE_DENSE_PREPROCESS else original_img

    full_results, cfg_used, pred_error, full_had_oom = predict_with_retry(model, working_img, mode="full")
    base_instances = []
    if full_results is not None and len(full_results) > 0:
        full_det = full_results[0]
        base_instances = extract_instances_from_result(full_det, offset_x=0, offset_y=0)
    else:
        print(f"  full-image inference returned no detections: {pred_error}")

    tile_instances = []
    tile_had_oom = False
    force_cpu = full_had_oom or is_oom_text(pred_error)

    if ENABLE_TILE_PASS:
        tile_instances, tile_had_oom = predict_tiles(model, working_img, cpu_only=force_cpu)

    if not tile_instances and (force_cpu or tile_had_oom):
        print("  switching to CPU fallback tile pass")
        cpu_tile_instances, _ = predict_tiles(model, working_img, cpu_only=True)
        tile_instances.extend(cpu_tile_instances)

    merged_instances = merge_instances(
        base_instances + tile_instances,
        iou_threshold=MERGE_IOU_THRESHOLD,
        class_agnostic=True,
    )

    if ENABLE_MASK_REFINEMENT:
        merged_instances = refine_instances(merged_instances)

    if not merged_instances:
        raise RuntimeError("no detections from full pass or tile pass")

    return {
        "original_img": original_img,
        "merged_instances": merged_instances,
        "full_count": len(base_instances),
        "tile_count": len(tile_instances),
        "cfg_used": cfg_used,
    }


def calculate_actual_area_um2(instance, source_shape_hw):
    source_h, source_w = source_shape_hw
    if source_h <= 0 or source_w <= 0:
        return 0.0

    pixel_to_um_w = ACTUAL_WIDTH_UM / source_w
    pixel_to_um_h = ACTUAL_HEIGHT_UM / source_h
    avg_conversion = (pixel_to_um_w + pixel_to_um_h) / 2.0

    pixel_area = float(instance.area_pixels)
    return pixel_area * (avg_conversion ** 2)


def process_folder(folder_path, model):
    root_output = os.path.join(folder_path, "results")
    if os.path.exists(root_output):
        try:
            shutil.rmtree(root_output)
        except Exception as exc:
            print(f"warning: could not delete old results dir {root_output}: {exc}")
    os.makedirs(root_output, exist_ok=True)

    summary_data = []
    failed_data = []
    used_sheet_names = set()

    pattern = r"CH(\d+).*IMG001x0(\d+)"
    match = re.search(pattern, folder_path.replace("/", "\\"))
    if match:
        excel_path = os.path.join(root_output, f"CH{int(match.group(1))}_CB{int(match.group(2))}.xlsx")
    else:
        excel_path = os.path.join(root_output, "results.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
        images = set()
        for ext in image_extensions:
            images.update(glob.glob(os.path.join(folder_path, f"*{ext}")))
            images.update(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        images = sorted(images, key=extract_number)

        if not images:
            print(f"warning: no images found in {folder_path}")
            pd.DataFrame([{"信息": f"文件夹 {os.path.basename(folder_path)} 中未找到图片"}]).to_excel(
                writer, sheet_name="无检测结果", index=False
            )
            return

        print(f"processing folder: {folder_path}, images={len(images)}")

        for image_path in tqdm(images):
            img_name = os.path.basename(image_path)
            sheet_name = make_unique_sheet_name(img_name, used_sheet_names)

            try:
                proc = process_image(model, image_path)
                original_img = proc["original_img"]
                merged_instances = proc["merged_instances"]

                if not merged_instances:
                    append_summary_row(summary_data, img_name, "未检出", 0, "0.00", "")
                    continue

                target_details = []
                total_area = 0.0

                for i, inst in enumerate(merged_instances, start=1):
                    area_um2 = calculate_actual_area_um2(inst, original_img.shape[:2])
                    total_area += area_um2
                    target_details.append(
                        {
                            "目标编号": i,
                            "类别ID": inst.cls,
                            "置信度": f"{inst.conf:.2f}",
                            "实际面积(μm²)": f"{area_um2:.2f}",
                        }
                    )

                mask_only = overlay_instances(original_img, merged_instances, show_labels=False, show_boxes=False)
                mask_with_ids = overlay_instances(original_img, merged_instances, show_labels=True, show_boxes=False)
                stats_lines = [
                    f"微藻总数: {len(merged_instances)}",
                    f"微藻总面积: {total_area:.2f} μm²",
                ]
                mask_with_ids = add_stats_panel(mask_with_ids, stats_lines)

                max_height = max(original_img.shape[0], mask_only.shape[0], mask_with_ids.shape[0])
                original_img_resized = cv2.resize(original_img, (original_img.shape[1], max_height))
                mask_only_resized = cv2.resize(mask_only, (mask_only.shape[1], max_height))
                mask_with_ids_resized = cv2.resize(mask_with_ids, (mask_with_ids.shape[1], max_height))
                combined_img = cv2.hconcat([original_img_resized, mask_only_resized, mask_with_ids_resized])

                output_path = os.path.join(root_output, f"R_{img_name}")
                cv2.imwrite(output_path, combined_img)
                print(f"saved: {output_path}")

                details_df = pd.DataFrame(target_details)
                details_df.to_excel(writer, sheet_name=sheet_name, index=False)

                stats_df = pd.DataFrame(
                    [
                        {
                            "图片名称": img_name,
                            "原始尺寸(像素)": f"{original_img.shape[1]}x{original_img.shape[0]}",
                            "目标总数": len(merged_instances),
                            "总面积(μm²)": f"{total_area:.2f}",
                        }
                    ]
                )
                stats_df.to_excel(writer, sheet_name=sheet_name, startrow=len(details_df) + 3, index=False)

                append_summary_row(summary_data, img_name, "成功", len(merged_instances), f"{total_area:.2f}", "")

            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}"
                print(f"error processing {image_path}: {err_text}")
                failed_data.append({"图片名称": img_name, "错误信息": err_text})
                append_summary_row(summary_data, img_name, "失败", 0, "0.00", err_text)
            finally:
                safe_empty_cuda_cache()
                gc.collect()

        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="汇总统计", index=False)
            print(f"saved summary: {excel_path}")

        if failed_data:
            pd.DataFrame(failed_data).to_excel(writer, sheet_name="失败记录", index=False)

        if not summary_data:
            pd.DataFrame([{"信息": f"文件夹 {os.path.basename(folder_path)} 中所有图片均未检出目标"}]).to_excel(
                writer, sheet_name="无检测结果", index=False
            )

    print(f"folder done: {folder_path}")


def process_root_folder(root_folder, model):
    if not os.path.isdir(root_folder):
        print(f"warning: root folder not found, skipped: {root_folder}")
        return

    for subdir in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subfolder_path):
            try:
                process_folder(subfolder_path, model)
            except Exception as exc:
                print(f"error processing folder {subfolder_path}: {type(exc).__name__}: {exc}")
                safe_empty_cuda_cache()
                gc.collect()


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model file not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for root_folder in ROOT_FOLDERS:
        process_root_folder(root_folder, model)

    print("done")


if __name__ == "__main__":
    main()
