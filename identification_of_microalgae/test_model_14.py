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


MODEL_PATH = r"E:\pythonProject\Microalgae_Identification_YOLOv11\runs\segment\train3\weights\best.pt"

# Update this list if needed.
ROOT_FOLDERS = [
    r"F:\Microalgae_Photoes\text_photoes\CH1",
]

ACTUAL_WIDTH_UM = 44.3
ACTUAL_HEIGHT_UM = 42.8

DENSE_MODE = True
ENABLE_TILE_PASS = True
ENABLE_DENSE_PREPROCESS = True
ENABLE_MASK_REFINEMENT = True
ENABLE_CHAMBER_FILTER = True

TILE_SIZE = 1024
TILE_OVERLAP = 256

MERGE_IOU_THRESHOLD = 0.35
MIN_MASK_AREA = 10
BOUNDARY_DARK_THRESHOLD = 120
BOUNDARY_ERODE_KERNEL = 7
MIN_INSIDE_RATIO = 0.30
CHAMBER_TEMPLATE_NAMES = ("chamber_template.png", "chamber_mask.png")
CHAMBER_MASK_DILATION = 0
MASK_REPAIR_MAX_KERNEL = 7
MASK_REPAIR_MAX_AREA_GAIN_RATIO = 0.18
MASK_REPAIR_MAX_AREA_LOSS_RATIO = 0.08
MASK_REPAIR_MAX_HOLE_AREA_RATIO = 0.12
MASK_REPAIR_MIN_IOU = 0.78
CHAMBER_CENTER_SEARCH_RADIUS = 64
CHAMBER_BORDER_MAX_RATIO = 0.05
ENABLE_TINY_INSTANCE_FILTER = True
TINY_INSTANCE_ABS_AREA = 450
TINY_INSTANCE_REL_AREA = 0.18
TINY_INSTANCE_MIN_SIDE = 4
TINY_INSTANCE_MIN_FILL_RATIO = 0.22
TINY_INSTANCE_MAX_ASPECT_RATIO = 6.0
TINY_INSTANCE_MIN_EQUIV_DIAMETER = 4.5
CHAMBER_CORE_ERODE_KERNEL = 9
CHAMBER_BORDER_TOUCH_MIN_INSIDE_RATIO = 0.50

FONT = cv2.FONT_HERSHEY_SIMPLEX
CPU_MODEL = None
CHAMBER_VISUAL_BOUNDARY_KERNEL = 5
CHAMBER_VISUAL_BOUNDARY_ALPHA = 0.85
CHAMBER_VISUAL_BOUNDARY_COLOR = (255, 0, 255)

BACKGROUND_SAMPLE_COUNT = 13
BACKGROUND_PERCENTILE = 50.0
BACKGROUND_BLUR_KERNEL = 3

SEARCH_IMAGE_SCALE = 0.35
SEARCH_DX_RANGE = 100
SEARCH_DY_RANGE = 140
COARSE_DX_STEP = 16
COARSE_DY_STEP = 16
REFINE_DX_RADIUS = 20
REFINE_DY_RADIUS = 20
REFINE_DX_STEP = 4
REFINE_DY_STEP = 4
FINAL_DX_RADIUS = 8
FINAL_DY_RADIUS = 8
FINAL_DX_STEP = 2
FINAL_DY_STEP = 2

COARSE_SCALE_VALUES = (0.90, 0.96, 1.00, 1.04, 1.10)
REFINE_SCALE_OFFSETS = (-0.03, -0.015, 0.0, 0.015, 0.03)
FINAL_SCALE_OFFSETS = (-0.01, 0.0, 0.01)

TEMPLATE_BOUNDARY_KERNEL = 5
INNER_CORE_KERNEL = 11
OUTER_RING_KERNEL = 11
DISTANCE_SIGMA_SMALL = 4.0
DISTANCE_SIGMA_FULL = 8.0
MASK_COVERAGE_TOLERANCE = 0.18
MIN_SHARED_SCORE = 0.22
LOCAL_PATCH_MAX_LEFT_GAP = 120
LOCAL_PATCH_MAX_BOTTOM_GAP = 120
LOCAL_PATCH_MAX_AREA_GAIN_RATIO = 0.25
LOCAL_PATCH_MIN_ROI_SIZE = 140
LOCAL_PATCH_MAX_ROI_SIZE = 320

SNAP_DISTANCE_SIGMA = 6.0
SNAP_SMOOTH_WINDOW = 15
SNAP_IOU_MIN = 0.94
SNAP_MAX_AREA_GAIN_RATIO = 0.055
SNAP_MAX_AREA_LOSS_RATIO = 0.025
SNAP_MIN_TOTAL_GAIN = 0.003
SNAP_MIN_BOTTOM_MID_GAIN = 0.002
SNAP_MIN_SIDE_GAIN = 0.0015

BOTTOM_REGION_Y_MIN = 0.68
BOTTOM_MID_X_MIN = 0.28
BOTTOM_MID_X_MAX = 0.76
LEFT_REGION_X_MAX = 0.28
RIGHT_REGION_X_MIN = 0.76
TOP_REGION_Y_MAX = 0.22

SIDE_SCORE_TOP_WEIGHT = 0.08
SIDE_SCORE_RIGHT_WEIGHT = 0.10
SIDE_SCORE_BOTTOM_WEIGHT = 0.10
SIDE_SCORE_LEFT_WEIGHT = 0.06
SIDE_SCORE_BOTTOM_MID_WEIGHT = 0.13
SIDE_SCORE_DISTANCE_WEIGHT = 0.23
SIDE_SCORE_BOUNDARY_WEIGHT = 0.15
SIDE_SCORE_CONTRAST_IN_WEIGHT = 0.05
SIDE_SCORE_CONTRAST_OUT_WEIGHT = 0.03

BOTTOM_ARC_X_MIN = 0.30
BOTTOM_ARC_X_MAX = 0.82
BOTTOM_ARC_Y_MIN = 0.58
BOTTOM_ARC_OFFSETS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
BOTTOM_ARC_OUTWARD_PRIOR = 0.005
BOTTOM_ARC_SMOOTH_WINDOW = 17
BOTTOM_ARC_IOU_MIN = 0.975
BOTTOM_ARC_MAX_AREA_GAIN_RATIO = 0.020
BOTTOM_ARC_MAX_AREA_LOSS_RATIO = 0.010
BOTTOM_ARC_MIN_SUPPORT_GAIN = 0.004
BOTTOM_ARC_MIN_MOVED_RATIO = 0.12
BOTTOM_ARC_MIN_OFFSET_GAIN = 0.006

TIGHT_MID_X_MIN = 0.40
TIGHT_MID_X_MAX = 0.80
TIGHT_MID_Y_MIN = 0.55
TIGHT_MID_OFFSETS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
TIGHT_MID_OUTWARD_PRIOR = 0.006
TIGHT_MID_MIN_OFFSET_GAIN = 0.004
TIGHT_MID_MIN_MOVED_RATIO = 0.10
TIGHT_MID_SMOOTH_WINDOW = 11
TIGHT_MID_IOU_MIN = 0.950
TIGHT_MID_MAX_AREA_GAIN_RATIO = 0.02
TIGHT_MID_MAX_AREA_LOSS_RATIO = 0.01
TIGHT_MID_MIN_SUPPORT_GAIN = 0.0025


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


def find_center_seed(free_mask):
    h, w = free_mask.shape[:2]
    cx = w // 2
    cy = h // 2

    if free_mask[cy, cx]:
        return cx, cy

    limit = min(CHAMBER_CENTER_SEARCH_RADIUS, max(h, w))
    for radius in range(1, limit + 1):
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        window = free_mask[y1:y2, x1:x2]
        ys, xs = np.where(window)
        if xs.size == 0:
            continue

        xs = xs + x1
        ys = ys + y1
        distances = (xs - cx) ** 2 + (ys - cy) ** 2
        idx = int(np.argmin(distances))
        return int(xs[idx]), int(ys[idx])

    return None


def mask_border_ratio(mask):
    if mask.size == 0:
        return 1.0
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(border.mean()) if border.size else 1.0


def build_chamber_from_seed(wall_mask):
    free_mask = ~wall_mask
    seed = find_center_seed(free_mask)
    if seed is None:
        return None

    seed_x, seed_y = seed
    num_labels, labels = cv2.connectedComponents(free_mask.astype(np.uint8), connectivity=8)
    seed_label = int(labels[seed_y, seed_x])
    if seed_label == 0:
        return None

    chamber_mask = labels == seed_label
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOUNDARY_ERODE_KERNEL, BOUNDARY_ERODE_KERNEL))
    chamber_mask = cv2.erode((chamber_mask.astype(np.uint8) * 255), kernel, iterations=1).astype(bool)

    coverage = float(chamber_mask.mean())
    if coverage < 0.02 or coverage > 0.98:
        return None
    if not chamber_mask[seed_y, seed_x]:
        return None
    if mask_border_ratio(chamber_mask) > CHAMBER_BORDER_MAX_RATIO:
        return None

    return chamber_mask


def build_chamber_from_corners(wall_mask):
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
    if mask_border_ratio(chamber_mask) > CHAMBER_BORDER_MAX_RATIO:
        return None

    return chamber_mask


def build_auto_chamber_mask(image_bgr):
    wall_mask = detect_black_wall_mask(image_bgr)
    chamber_mask = build_chamber_from_seed(wall_mask)
    if chamber_mask is not None:
        return chamber_mask

    return build_chamber_from_corners(wall_mask)


def build_chamber_core_mask(chamber_mask, erode_kernel=CHAMBER_CORE_ERODE_KERNEL):
    if chamber_mask is None:
        return None

    kernel_size = max(3, int(erode_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    core = cv2.erode(chamber_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0

    if core.sum() == 0:
        return chamber_mask

    return core


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


def instance_inside_chamber(instance, chamber_mask, chamber_core_mask=None):
    x1, y1, x2, y2 = instance.box
    if x2 <= x1 or y2 <= y1:
        return False

    chamber_roi = chamber_mask[y1:y2, x1:x2]
    if chamber_roi.shape != instance.mask.shape:
        chamber_roi = resize_mask_to_shape(chamber_roi, instance.mask.shape[:2])

    core_roi = None
    if chamber_core_mask is not None:
        core_roi = chamber_core_mask[y1:y2, x1:x2]
        if core_roi.shape != instance.mask.shape:
            core_roi = resize_mask_to_shape(core_roi, instance.mask.shape[:2])

    total_pixels = int(instance.mask.sum())
    if total_pixels <= 0:
        return False

    inside_pixels = int(np.logical_and(instance.mask, chamber_roi).sum())
    inside_ratio = inside_pixels / total_pixels

    core_pixels = 0
    boundary_band_roi = None
    if core_roi is not None:
        core_pixels = int(np.logical_and(instance.mask, core_roi).sum())
        boundary_band_roi = np.logical_and(chamber_roi, np.logical_not(core_roi))
    touches_boundary_band = bool(np.logical_and(instance.mask, boundary_band_roi).any()) if boundary_band_roi is not None else False

    ys, xs = np.where(instance.mask)
    if xs.size == 0 or ys.size == 0:
        return False
    cx = x1 + int(np.mean(xs))
    cy = y1 + int(np.mean(ys))
    if cy < 0 or cx < 0 or cy >= chamber_mask.shape[0] or cx >= chamber_mask.shape[1]:
        return False

    centroid_in_chamber = bool(chamber_mask[cy, cx])
    if not centroid_in_chamber:
        return False

    if core_roi is not None and touches_boundary_band:
        return core_pixels > 0 and inside_ratio >= CHAMBER_BORDER_TOUCH_MIN_INSIDE_RATIO

    return inside_ratio >= MIN_INSIDE_RATIO


def filter_instances_inside_chamber(instances, chamber_mask, chamber_core_mask=None):
    if not instances or chamber_mask is None:
        return instances

    filtered = [inst for inst in instances if instance_inside_chamber(inst, chamber_mask, chamber_core_mask)]
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


def get_instance_geometry(instance):
    x1, y1, x2, y2 = instance.box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = max(1, box_w * box_h)
    area = int(instance.area_pixels)
    min_side = min(box_w, box_h)
    max_side = max(box_w, box_h)
    fill_ratio = float(area) / float(box_area)
    aspect_ratio = float(max_side) / float(max(1, min_side))
    equiv_diameter = float(np.sqrt((4.0 * float(area)) / np.pi)) if area > 0 else 0.0
    return {
        "area": area,
        "box_w": box_w,
        "box_h": box_h,
        "min_side": min_side,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "equiv_diameter": equiv_diameter,
    }


def filter_tiny_instances(instances):
    if not instances:
        return instances

    areas = np.asarray([inst.area_pixels for inst in instances], dtype=np.float32)
    median_area = float(np.median(areas)) if areas.size > 0 else 0.0
    area_threshold = max(TINY_INSTANCE_ABS_AREA, int(round(median_area * TINY_INSTANCE_REL_AREA)))

    kept = []
    removed = 0
    for inst in instances:
        geom = get_instance_geometry(inst)
        area = geom["area"]

        if area >= area_threshold:
            kept.append(inst)
            continue

        tiny_like = (
            area <= TINY_INSTANCE_ABS_AREA
            or geom["min_side"] <= TINY_INSTANCE_MIN_SIDE
            or geom["fill_ratio"] <= TINY_INSTANCE_MIN_FILL_RATIO
            or geom["aspect_ratio"] >= TINY_INSTANCE_MAX_ASPECT_RATIO
            or geom["equiv_diameter"] <= TINY_INSTANCE_MIN_EQUIV_DIAMETER
        )

        if tiny_like:
            removed += 1
        else:
            kept.append(inst)

    if removed > 0:
        print(f"  tiny filter removed {removed} instance(s); area_threshold={area_threshold}")

    return kept


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

    wall_mask = detect_black_wall_mask(original_img)
    chamber_mask = build_auto_chamber_mask(original_img)
    chamber_core_mask = build_chamber_core_mask(chamber_mask)
    template_mask = load_template_chamber_mask(image_path, original_img.shape[:2])
    if template_mask is not None and wall_mask is not None:
        template_mask = align_template_to_wall(template_mask, wall_mask)

    if ENABLE_CHAMBER_FILTER:
        if chamber_mask is not None and CHAMBER_MASK_DILATION > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (CHAMBER_MASK_DILATION, CHAMBER_MASK_DILATION),
            )
            chamber_mask = cv2.dilate(chamber_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        elif template_mask is not None:
            chamber_mask = template_mask
            chamber_core_mask = build_chamber_core_mask(chamber_mask)

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

    if ENABLE_CHAMBER_FILTER:
        base_instances = filter_instances_inside_chamber(base_instances, chamber_mask, chamber_core_mask)
        tile_instances = filter_instances_inside_chamber(tile_instances, chamber_mask, chamber_core_mask)

    merged_instances = merge_instances(
        base_instances + tile_instances,
        iou_threshold=MERGE_IOU_THRESHOLD,
        class_agnostic=True,
    )

    if ENABLE_CHAMBER_FILTER:
        merged_instances = filter_instances_inside_chamber(merged_instances, chamber_mask, chamber_core_mask)

    if ENABLE_MASK_REFINEMENT:
        merged_instances = refine_instances(merged_instances)

    if ENABLE_TINY_INSTANCE_FILTER:
        merged_instances = filter_tiny_instances(merged_instances)

    if not merged_instances:
        raise RuntimeError("no detections from full pass or tile pass")

    return {
        "original_img": original_img,
        "merged_instances": merged_instances,
        "full_count": len(base_instances),
        "tile_count": len(tile_instances),
        "cfg_used": cfg_used,
        "chamber_mask": chamber_mask if ENABLE_CHAMBER_FILTER else None,
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


REFERENCE_TEMPLATE_NAMES = (
    "chamber_template.png",
    "chamber_mask.png",
    "chamber_template2.png",
    "chamber_mask2.png",
)

CURRENT_SHARED_CHAMBER_MASK = None
CURRENT_SHARED_CHAMBER_CORE_MASK = None
CURRENT_SHARED_DEBUG = None
ORIGINAL_PROCESS_IMAGE = process_image
ORIGINAL_PROCESS_FOLDER = process_folder


def overlay_chamber_boundary(
    image_bgr,
    chamber_mask,
    alpha=CHAMBER_VISUAL_BOUNDARY_ALPHA,
    boundary_kernel=CHAMBER_VISUAL_BOUNDARY_KERNEL,
    boundary_color=CHAMBER_VISUAL_BOUNDARY_COLOR,
):
    if chamber_mask is None:
        return image_bgr

    mask = chamber_mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return image_bgr

    kernel_size = max(3, int(boundary_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask_u8 = mask.astype(np.uint8) * 255
    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel) > 0
    if not boundary.any():
        return image_bgr

    canvas = image_bgr.copy()
    color_arr = np.array(boundary_color, dtype=np.uint8)
    canvas[boundary] = (canvas[boundary] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    return canvas


def collect_folder_images(folder_path):
    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    images = set()
    for ext in image_extensions:
        images.update(glob.glob(os.path.join(folder_path, f"*{ext}")))
        images.update(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(images, key=extract_number)


def sample_images_uniformly(images, sample_count):
    if len(images) <= sample_count:
        return list(images)
    if sample_count <= 1:
        return [images[len(images) // 2]]

    indices = [int(round(i * (len(images) - 1) / float(sample_count - 1))) for i in range(sample_count)]
    return [images[idx] for idx in indices]


def build_sampled_background(image_paths):
    frames = []
    for image_path in image_paths:
        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            print(f"  warning: failed to read sampled image: {image_path}")
            continue
        frames.append(image_bgr.astype(np.uint8))

    if not frames:
        return None

    stack = np.stack(frames, axis=0)
    background = np.percentile(stack, BACKGROUND_PERCENTILE, axis=0).astype(np.uint8)

    kernel_size = ensure_odd(BACKGROUND_BLUR_KERNEL)
    if kernel_size >= 3:
        background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)

    return background


def load_reference_template_mask(reference_shape):
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(),
        os.getcwd(),
    ]

    for base_dir in search_dirs:
        for name in REFERENCE_TEMPLATE_NAMES:
            mask_path = os.path.join(base_dir, name)
            if not os.path.exists(mask_path):
                continue
            try:
                mask_img = Image.open(mask_path).convert("L")
                mask = np.asarray(mask_img)
                if mask.shape[:2] != reference_shape:
                    mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_NEAREST)
                print(f"  reference template loaded: {mask_path}")
                return mask > 127
            except Exception as exc:
                print(f"  warning: failed to load template {mask_path}: {type(exc).__name__}: {exc}")

    return None


def robust_normalize(values, low_q=50.0, high_q=99.5):
    values = np.asarray(values, dtype=np.float32)
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def build_response_maps(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=1.1)

    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    dark_norm = robust_normalize(255.0 - blur, low_q=55.0, high_q=99.5)
    grad_norm = robust_normalize(grad_mag, low_q=70.0, high_q=99.5)
    response = 0.60 * grad_norm + 0.40 * dark_norm
    response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=0.8)

    canny = cv2.Canny(blur, 35, 110) > 0
    strong_edge = np.logical_and(dark_norm > 0.20, grad_norm > 0.15)
    edge = np.logical_or(canny, strong_edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(edge.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    edge = cv2.dilate(edge.astype(np.uint8) * 255, kernel, iterations=1) > 0

    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)

    return {
        "gray": gray_eq,
        "response": response.astype(np.float32),
        "edge": edge,
        "distance": distance.astype(np.float32),
        "shape": gray_eq.shape[:2],
    }


def build_boundary_band(mask, kernel_size=TEMPLATE_BOUNDARY_KERNEL):
    kernel_size = ensure_odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = mask.astype(np.uint8) * 255
    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel) > 0
    if boundary.any():
        boundary = cv2.dilate(boundary.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return boundary


def build_inner_core(mask, kernel_size=INNER_CORE_KERNEL):
    kernel_size = ensure_odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return eroded if eroded.any() else mask


def build_outer_ring(mask, kernel_size=OUTER_RING_KERNEL):
    kernel_size = ensure_odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return np.logical_and(dilated, np.logical_not(mask))


def build_affine_matrix(shape_hw, sx, sy, dx, dy):
    h, w = shape_hw
    ax = (w - 1) * 0.5
    ay = (h - 1) * 0.5
    return np.array(
        [
            [sx, 0.0, dx + (1.0 - sx) * ax],
            [0.0, sy, dy + (1.0 - sy) * ay],
        ],
        dtype=np.float32,
    )


def warp_bool_mask_affine(mask, sx, sy, dx, dy, out_shape):
    matrix = build_affine_matrix(out_shape, sx, sy, dx, dy)
    warped = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (out_shape[1], out_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped > 0


def clip_scale(value):
    return float(np.clip(value, 0.84, 1.16))


def iter_values_around(center, offsets):
    values = []
    for offset in offsets:
        value = clip_scale(center + offset)
        if value not in values:
            values.append(value)
    return values


def compute_mask_border_stats(mask):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None

    h, w = mask.shape[:2]
    left_gap = int(xs.min())
    right_gap = int((w - 1) - xs.max())
    top_gap = int(ys.min())
    bottom_gap = int((h - 1) - ys.max())
    left_touch = float(mask[:, 0].mean()) if w > 0 else 0.0
    bottom_touch = float(mask[-1, :].mean()) if h > 0 else 0.0

    return {
        "left_gap": left_gap,
        "right_gap": right_gap,
        "top_gap": top_gap,
        "bottom_gap": bottom_gap,
        "left_touch": left_touch,
        "bottom_touch": bottom_touch,
    }


def mask_bbox(mask):
    return mask_to_bbox(mask)


def translate_bool_mask(mask, dx, dy):
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return warped > 0


def build_local_corner_weight(shape_hw, x_limit, y_start):
    h, w = shape_hw
    yy, xx = np.indices((h, w), dtype=np.float32)

    x_limit = max(1, int(x_limit))
    y_start = int(y_start)
    y_denom = max(1.0, float((h - 1) - y_start))

    wx = np.clip((float(x_limit) - xx) / float(x_limit), 0.0, 1.0)
    wy = np.clip((yy - float(y_start)) / y_denom, 0.0, 1.0)
    weight = wx * wy
    weight[: max(0, y_start), :] = 0.0
    weight[:, min(w, x_limit):] = 0.0
    return weight


def apply_local_left_bottom_patch(mask):
    stats = compute_mask_border_stats(mask)
    bbox = mask_bbox(mask)
    if stats is None or bbox is None:
        return mask, {"patch_applied": False, "patch_reason": "invalid_mask"}

    left_gap = int(stats["left_gap"])
    bottom_gap = int(stats["bottom_gap"])
    if left_gap <= 0 and bottom_gap <= 0:
        return mask, {"patch_applied": False, "patch_reason": "already_touching"}

    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    h, w = mask.shape[:2]

    roi_w = max(LOCAL_PATCH_MIN_ROI_SIZE, min(LOCAL_PATCH_MAX_ROI_SIZE, int(round(bbox_w * 0.30))))
    roi_h = max(LOCAL_PATCH_MIN_ROI_SIZE, min(LOCAL_PATCH_MAX_ROI_SIZE, int(round(bbox_h * 0.30))))
    x_limit = min(w, max(x1 + left_gap + 40, roi_w))
    y_start = max(0, min(h - 1, y2 - roi_h))

    weight = build_local_corner_weight(mask.shape[:2], x_limit, y_start)
    xx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    map_x = np.clip(xx + float(left_gap) * weight, 0.0, float(w - 1)).astype(np.float32)
    map_y = np.clip(yy - float(bottom_gap) * weight, 0.0, float(h - 1)).astype(np.float32)

    remapped = cv2.remap(
        mask.astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[y_start:, :x_limit] = True
    shifted_corner = translate_bool_mask(mask, dx=-left_gap, dy=bottom_gap)
    patched = np.logical_or(mask, np.logical_and(remapped, roi_mask))
    patched = np.logical_or(patched, np.logical_and(shifted_corner, roi_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    patched_roi = cv2.morphologyEx((patched.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    patched = np.logical_or(mask, np.logical_and(patched_roi, roi_mask))

    patched_stats = compute_mask_border_stats(patched)
    if patched_stats is None:
        return mask, {"patch_applied": False, "patch_reason": "patched_invalid"}

    original_area = int(mask.sum())
    patched_area = int(patched.sum())
    area_gain_ratio = (patched_area - original_area) / float(max(1, original_area))
    center_ok = bool(patched[h // 2, w // 2])
    improved = (
        patched_stats["left_gap"] <= stats["left_gap"]
        and patched_stats["bottom_gap"] <= stats["bottom_gap"]
        and (patched_stats["left_gap"] < stats["left_gap"] or patched_stats["bottom_gap"] < stats["bottom_gap"])
    )

    if not center_ok:
        return mask, {"patch_applied": False, "patch_reason": "center_lost"}
    if area_gain_ratio > LOCAL_PATCH_MAX_AREA_GAIN_RATIO:
        return mask, {"patch_applied": False, "patch_reason": "area_gain_too_large"}
    if not improved:
        return mask, {"patch_applied": False, "patch_reason": "no_gap_improvement"}

    return patched, {
        "patch_applied": True,
        "patch_reason": "ok",
        "patch_area_gain_ratio": float(area_gain_ratio),
        "patch_left_gap_before": int(stats["left_gap"]),
        "patch_bottom_gap_before": int(stats["bottom_gap"]),
        "patch_left_gap_after": int(patched_stats["left_gap"]),
        "patch_bottom_gap_after": int(patched_stats["bottom_gap"]),
        "patch_roi_width": int(x_limit),
        "patch_roi_height": int(h - y_start),
    }


def force_left_bottom_anchor(mask):
    stats = compute_mask_border_stats(mask)
    if stats is None:
        return mask, {"anchor_applied": False, "anchor_reason": "invalid_mask"}

    left_gap = int(stats["left_gap"])
    bottom_gap = int(stats["bottom_gap"])
    if left_gap <= 0 and bottom_gap <= 0:
        return mask, {"anchor_applied": False, "anchor_reason": "already_touching"}

    anchored = translate_bool_mask(mask, dx=-left_gap, dy=bottom_gap)
    anchored_stats = compute_mask_border_stats(anchored)
    if anchored_stats is None:
        return mask, {"anchor_applied": False, "anchor_reason": "anchored_invalid"}

    h, w = mask.shape[:2]
    if not anchored[h // 2, w // 2]:
        return mask, {"anchor_applied": False, "anchor_reason": "center_lost"}

    return anchored, {
        "anchor_applied": True,
        "anchor_reason": "ok",
        "anchor_left_gap_before": int(left_gap),
        "anchor_bottom_gap_before": int(bottom_gap),
        "anchor_left_gap_after": int(anchored_stats["left_gap"]),
        "anchor_bottom_gap_after": int(anchored_stats["bottom_gap"]),
    }


def evaluate_candidate(template_mask, maps, template_coverage, sx, sy, dx_full, dy_full, dx_range_full, dy_range_full, sigma):
    scale_factor = float(maps["scale_factor"])
    dx = float(dx_full) * scale_factor
    dy = float(dy_full) * scale_factor

    mask = warp_bool_mask_affine(template_mask, sx, sy, dx, dy, maps["shape"])
    if not mask.any():
        return None

    h, w = mask.shape[:2]
    cy = h // 2
    cx = w // 2
    if not mask[cy, cx]:
        return None

    coverage = float(mask.mean())
    if coverage < max(0.12, template_coverage - MASK_COVERAGE_TOLERANCE):
        return None
    if coverage > min(0.90, template_coverage + MASK_COVERAGE_TOLERANCE):
        return None

    border_stats = compute_mask_border_stats(mask)
    if border_stats is None:
        return None

    boundary = build_boundary_band(mask)
    if int(boundary.sum()) < 60:
        return None

    inner_core = build_inner_core(mask)
    outer_ring = build_outer_ring(mask)

    response_map = maps["response"]
    distance_map = maps["distance"]

    boundary_response = float(response_map[boundary].mean()) if boundary.any() else 0.0
    distance_score = float(np.exp(-distance_map[boundary] / sigma).mean()) if boundary.any() else 0.0
    inner_response = float(response_map[inner_core].mean()) if inner_core.any() else boundary_response
    outer_response = float(response_map[outer_ring].mean()) if outer_ring.any() else 0.0

    contrast_inside = boundary_response - inner_response
    contrast_outside = boundary_response - outer_response

    translation_prior = 1.0 - 0.5 * (
        abs(float(dx_full)) / max(1.0, float(dx_range_full)) +
        abs(float(dy_full)) / max(1.0, float(dy_range_full))
    )
    scale_prior = 1.0 - 0.5 * ((abs(sx - 1.0) / 0.16) + (abs(sy - 1.0) / 0.16))

    score = (
        0.45 * distance_score +
        0.25 * boundary_response +
        0.10 * max(0.0, contrast_inside) +
        0.10 * max(0.0, contrast_outside) +
        0.05 * max(0.0, translation_prior) +
        0.05 * max(0.0, scale_prior)
    )

    return {
        "score": float(score),
        "coverage": coverage,
        "boundary_response": boundary_response,
        "distance_score": distance_score,
        "contrast_inside": float(contrast_inside),
        "contrast_outside": float(contrast_outside),
        "translation_prior": float(translation_prior),
        "scale_prior": float(scale_prior),
        "sx": float(sx),
        "sy": float(sy),
        "dx_full": int(round(dx_full)),
        "dy_full": int(round(dy_full)),
        "left_gap": int(border_stats["left_gap"]),
        "right_gap": int(border_stats["right_gap"]),
        "top_gap": int(border_stats["top_gap"]),
        "bottom_gap": int(border_stats["bottom_gap"]),
        "left_touch": float(border_stats["left_touch"]),
        "bottom_touch": float(border_stats["bottom_touch"]),
        "mask": mask,
    }


def search_stage(template_mask, template_coverage, maps, dx_values, dy_values, sx_values, sy_values, dx_range_full, dy_range_full, sigma):
    best = None
    for sx in sx_values:
        for sy in sy_values:
            for dx_full in dx_values:
                for dy_full in dy_values:
                    candidate = evaluate_candidate(
                        template_mask,
                        maps,
                        template_coverage,
                        sx,
                        sy,
                        dx_full,
                        dy_full,
                        dx_range_full,
                        dy_range_full,
                        sigma,
                    )
                    if candidate is None:
                        continue
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate
    return best


def build_search_range(center, radius, step):
    center = int(round(center))
    radius = int(round(radius))
    step = max(1, int(round(step)))
    values = list(range(center - radius, center + radius + 1, step))
    if center not in values:
        values.append(center)
    return sorted(set(values))


def sample_float_map(map_2d, xs, ys):
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    sampled = cv2.remap(
        map_2d.astype(np.float32),
        xs.reshape(1, -1),
        ys.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1)


def largest_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 8:
        return None
    return contour[:, 0, :].astype(np.float32)


def mask_iou_bool(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def circular_smooth(values, window):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or window <= 1:
        return values
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        return np.full_like(values, float(np.mean(values)))
    radius = window // 2
    padded = np.concatenate([values[-radius:], values, values[:radius]])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def outward_normals(mask, contour):
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside - outside
    grad_x = cv2.Sobel(signed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(signed, cv2.CV_32F, 0, 1, ksize=3)

    xs = np.clip(contour[:, 0], 0, mask.shape[1] - 1)
    ys = np.clip(contour[:, 1], 0, mask.shape[0] - 1)
    inward_x = sample_float_map(grad_x, xs, ys)
    inward_y = sample_float_map(grad_y, xs, ys)

    outward = np.stack([-inward_x, -inward_y], axis=1)
    norms = np.linalg.norm(outward, axis=1, keepdims=True)

    centroid = np.mean(contour, axis=0, keepdims=True)
    radial = contour - centroid
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm[radial_norm < 1e-6] = 1.0
    radial = radial / radial_norm

    safe = norms[:, 0] > 1e-4
    outward[safe] = outward[safe] / norms[safe]
    outward[~safe] = radial[~safe]
    return outward.astype(np.float32)


def ensure_snap_support_map(maps, sigma):
    sigma_key = f"blackedge_support_sigma_{sigma:.2f}"
    if sigma_key in maps:
        return maps[sigma_key]

    dark_norm = robust_normalize(255.0 - maps["gray"], low_q=55.0, high_q=99.5)
    edge_term = np.exp(-maps["distance"] / float(sigma))
    ridge_term = dark_norm * edge_term
    support = (
        0.40 * maps["response"] +
        0.25 * dark_norm +
        0.35 * ridge_term
    ).astype(np.float32)
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=0.8)
    maps[sigma_key] = support
    return support


def masked_mean(values, mask, fallback):
    if mask is None or not np.any(mask):
        return float(fallback)
    return float(np.mean(values[mask]))


def build_boundary_side_index(mask, boundary):
    bbox = mask_bbox(mask)
    if bbox is None:
        return None

    ys, xs = np.where(boundary)
    if xs.size == 0 or ys.size == 0:
        return None

    x1, y1, x2, y2 = bbox
    x2 -= 1
    y2 -= 1
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    dist_top = ys - y1
    dist_right = x2 - xs
    dist_bottom = y2 - ys
    dist_left = xs - x1
    distances = np.stack([dist_top, dist_right, dist_bottom, dist_left], axis=1)
    side_idx = np.argmin(distances, axis=1)

    x_norm = (xs - x1) / float(max(1, bbox_w))
    y_norm = (ys - y1) / float(max(1, bbox_h))
    bottom_mid = np.logical_and.reduce(
        [
            side_idx == 2,
            x_norm >= BOTTOM_MID_X_MIN,
            x_norm <= BOTTOM_MID_X_MAX,
            y_norm >= BOTTOM_REGION_Y_MIN,
        ]
    )

    return {
        "ys": ys,
        "xs": xs,
        "side_idx": side_idx,
        "bottom_mid": bottom_mid,
        "bbox": bbox,
    }


def measure_mask_fit(mask, maps, sigma):
    boundary = build_boundary_band(mask)
    if not boundary.any():
        return None

    side_index = build_boundary_side_index(mask, boundary)
    if side_index is None:
        return None

    ys = side_index["ys"]
    xs = side_index["xs"]
    side_idx = side_index["side_idx"]
    bottom_mid = side_index["bottom_mid"]

    support_map = ensure_snap_support_map(maps, sigma)
    support_samples = support_map[ys, xs]
    distance_samples = np.exp(-maps["distance"][ys, xs] / float(sigma))

    boundary_response = float(np.mean(support_samples))
    distance_score = float(np.mean(distance_samples))

    top_score = masked_mean(support_samples, side_idx == 0, boundary_response)
    right_score = masked_mean(support_samples, side_idx == 1, boundary_response)
    bottom_score = masked_mean(support_samples, side_idx == 2, boundary_response)
    left_score = masked_mean(support_samples, side_idx == 3, boundary_response)
    bottom_mid_score = masked_mean(support_samples, bottom_mid, bottom_score)

    inner_core = build_inner_core(mask)
    outer_ring = build_outer_ring(mask)
    inner_response = float(maps["response"][inner_core].mean()) if inner_core.any() else boundary_response
    outer_response = float(maps["response"][outer_ring].mean()) if outer_ring.any() else boundary_response

    contrast_inside = boundary_response - inner_response
    contrast_outside = boundary_response - outer_response

    side_score = (
        SIDE_SCORE_DISTANCE_WEIGHT * distance_score +
        SIDE_SCORE_BOUNDARY_WEIGHT * boundary_response +
        SIDE_SCORE_TOP_WEIGHT * top_score +
        SIDE_SCORE_RIGHT_WEIGHT * right_score +
        SIDE_SCORE_BOTTOM_WEIGHT * bottom_score +
        SIDE_SCORE_LEFT_WEIGHT * left_score +
        SIDE_SCORE_BOTTOM_MID_WEIGHT * bottom_mid_score +
        SIDE_SCORE_CONTRAST_IN_WEIGHT * max(0.0, contrast_inside) +
        SIDE_SCORE_CONTRAST_OUT_WEIGHT * max(0.0, contrast_outside)
    )

    return {
        "boundary_response": boundary_response,
        "distance_score": distance_score,
        "top_score": top_score,
        "right_score": right_score,
        "bottom_score": bottom_score,
        "left_score": left_score,
        "bottom_mid_score": bottom_mid_score,
        "contrast_inside": float(contrast_inside),
        "contrast_outside": float(contrast_outside),
        "side_score": float(side_score),
    }


def contour_side_groups(contour, bbox):
    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))

    x_norm = (contour[:, 0] - float(x1)) / bbox_w
    y_norm = (contour[:, 1] - float(y1)) / bbox_h

    dist_top = y_norm
    dist_right = 1.0 - x_norm
    dist_bottom = 1.0 - y_norm
    dist_left = x_norm
    side_idx = np.argmin(np.stack([dist_top, dist_right, dist_bottom, dist_left], axis=1), axis=1)

    top = side_idx == 0
    right = side_idx == 1
    bottom = side_idx == 2
    left = side_idx == 3

    bottom_mid = np.logical_and.reduce(
        [
            bottom,
            x_norm >= BOTTOM_MID_X_MIN,
            x_norm <= BOTTOM_MID_X_MAX,
            y_norm >= BOTTOM_REGION_Y_MIN,
        ]
    )
    left_bottom = np.logical_and.reduce(
        [
            x_norm <= LEFT_REGION_X_MAX,
            y_norm >= 0.56,
        ]
    )
    right_mid = np.logical_and.reduce(
        [
            right,
            x_norm >= RIGHT_REGION_X_MIN,
            y_norm >= 0.22,
            y_norm <= 0.88,
        ]
    )
    top_arc = np.logical_and(top, y_norm <= TOP_REGION_Y_MAX)

    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
        "bottom_mid": bottom_mid,
        "left_bottom": left_bottom,
        "right_mid": right_mid,
        "top_arc": top_arc,
    }


def choose_offsets(contour, normals, maps, bbox):
    support = ensure_snap_support_map(maps, SNAP_DISTANCE_SIGMA)
    groups = contour_side_groups(contour, bbox)

    n = len(contour)
    offsets = np.zeros(n, dtype=np.float32)
    min_offsets = np.full(n, -2.0, dtype=np.float32)
    max_offsets = np.full(n, 4.0, dtype=np.float32)

    def solve(group_mask, offset_values, outward_penalty, inward_penalty, outward_bonus=0.0):
        idx = np.flatnonzero(group_mask)
        if idx.size == 0:
            return

        pts = contour[idx]
        nrms = normals[idx]
        best_offsets = np.zeros(idx.size, dtype=np.float32)
        best_scores = np.full(idx.size, -1e9, dtype=np.float32)

        for off in offset_values:
            xs = np.clip(pts[:, 0] + nrms[:, 0] * off, 0, maps["shape"][1] - 1)
            ys = np.clip(pts[:, 1] + nrms[:, 1] * off, 0, maps["shape"][0] - 1)
            values = sample_float_map(support, xs, ys)
            score = (
                values
                + float(outward_bonus) * max(0.0, float(off))
                - float(outward_penalty) * max(0.0, float(off))
                - float(inward_penalty) * max(0.0, -float(off))
            )
            better = score > best_scores
            best_scores[better] = score[better]
            best_offsets[better] = float(off)

        offsets[idx] = best_offsets
        min_offsets[idx] = float(min(offset_values))
        max_offsets[idx] = float(max(offset_values))

    remaining = np.ones(n, dtype=bool)

    solve(groups["bottom_mid"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], 0.008, 0.080, outward_bonus=0.0025)
    remaining &= ~groups["bottom_mid"]

    solve(np.logical_and(groups["left_bottom"], remaining), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0], 0.006, 0.045, outward_bonus=0.0040)
    remaining &= ~groups["left_bottom"]

    solve(np.logical_and(groups["bottom"], remaining), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], 0.011, 0.055, outward_bonus=0.0012)
    remaining &= ~groups["bottom"]

    solve(np.logical_and(groups["right_mid"], remaining), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 0.012, 0.040, outward_bonus=0.0008)
    remaining &= ~groups["right_mid"]

    solve(np.logical_and(groups["top_arc"], remaining), [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], 0.018, 0.030, outward_bonus=0.0)
    remaining &= ~groups["top_arc"]

    solve(np.logical_and(groups["left"], remaining), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 0.012, 0.050, outward_bonus=0.0010)
    remaining &= ~groups["left"]

    solve(np.logical_and(groups["right"], remaining), [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 0.014, 0.040, outward_bonus=0.0006)
    remaining &= ~groups["right"]

    solve(remaining, [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], 0.014, 0.040, outward_bonus=0.0005)

    smoothed = circular_smooth(offsets, SNAP_SMOOTH_WINDOW)
    smoothed = np.clip(smoothed, min_offsets, max_offsets)
    return smoothed.astype(np.float32)


def snap_mask_to_black_edge(mask, maps):
    contour = largest_contour(mask)
    bbox = mask_bbox(mask)
    if contour is None or bbox is None:
        return mask, {"edge_snap_applied": False, "edge_snap_reason": "missing_contour"}

    fit_before = measure_mask_fit(mask, maps, SNAP_DISTANCE_SIGMA)
    if fit_before is None:
        return mask, {"edge_snap_applied": False, "edge_snap_reason": "fit_before_invalid"}

    normals = outward_normals(mask, contour)
    offsets = choose_offsets(contour, normals, maps, bbox)

    snapped_points = contour + normals * offsets[:, None]
    snapped_points[:, 0] = np.clip(snapped_points[:, 0], 0, mask.shape[1] - 1)
    snapped_points[:, 1] = np.clip(snapped_points[:, 1], 0, mask.shape[0] - 1)

    snapped_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(snapped_mask, [np.round(snapped_points).astype(np.int32)], 255)
    snapped_mask = snapped_mask > 0

    if not snapped_mask[mask.shape[0] // 2, mask.shape[1] // 2]:
        return mask, {"edge_snap_applied": False, "edge_snap_reason": "center_lost"}

    iou = mask_iou_bool(mask, snapped_mask)
    if iou < SNAP_IOU_MIN:
        return mask, {"edge_snap_applied": False, "edge_snap_reason": "iou_too_low", "edge_snap_iou": float(iou)}

    original_area = int(mask.sum())
    snapped_area = int(snapped_mask.sum())
    area_gain_ratio = (snapped_area - original_area) / float(max(1, original_area))
    if area_gain_ratio > SNAP_MAX_AREA_GAIN_RATIO or area_gain_ratio < -SNAP_MAX_AREA_LOSS_RATIO:
        return mask, {
            "edge_snap_applied": False,
            "edge_snap_reason": "area_change_too_large",
            "edge_snap_area_gain_ratio": float(area_gain_ratio),
        }

    fit_after = measure_mask_fit(snapped_mask, maps, SNAP_DISTANCE_SIGMA)
    if fit_after is None:
        return mask, {"edge_snap_applied": False, "edge_snap_reason": "fit_after_invalid"}

    total_gain = fit_after["side_score"] - fit_before["side_score"]
    top_gain = fit_after["top_score"] - fit_before["top_score"]
    right_gain = fit_after["right_score"] - fit_before["right_score"]
    bottom_gain = fit_after["bottom_score"] - fit_before["bottom_score"]
    left_gain = fit_after["left_score"] - fit_before["left_score"]
    bottom_mid_gain = fit_after["bottom_mid_score"] - fit_before["bottom_mid_score"]

    positive_side_count = sum(
        gain >= SNAP_MIN_SIDE_GAIN
        for gain in (top_gain, right_gain, bottom_gain, left_gain, bottom_mid_gain)
    )
    if total_gain < SNAP_MIN_TOTAL_GAIN:
        return mask, {
            "edge_snap_applied": False,
            "edge_snap_reason": "total_gain_too_small",
            "edge_snap_total_gain": float(total_gain),
            "edge_snap_bottom_mid_gain": float(bottom_mid_gain),
        }
    if positive_side_count == 0 and bottom_mid_gain < SNAP_MIN_BOTTOM_MID_GAIN:
        return mask, {
            "edge_snap_applied": False,
            "edge_snap_reason": "local_gain_too_small",
            "edge_snap_total_gain": float(total_gain),
            "edge_snap_bottom_mid_gain": float(bottom_mid_gain),
        }

    return snapped_mask, {
        "edge_snap_applied": True,
        "edge_snap_reason": "ok",
        "edge_snap_iou": float(iou),
        "edge_snap_area_gain_ratio": float(area_gain_ratio),
        "edge_snap_total_gain": float(total_gain),
        "edge_snap_top_gain": float(top_gain),
        "edge_snap_right_gain": float(right_gain),
        "edge_snap_bottom_gain": float(bottom_gain),
        "edge_snap_left_gain": float(left_gain),
        "edge_snap_bottom_mid_gain": float(bottom_mid_gain),
        "edge_snap_mean_offset": float(np.mean(offsets)),
        "edge_snap_max_offset": float(np.max(np.abs(offsets))),
        "edge_snap_positive_side_count": int(positive_side_count),
    }


def bottom_arc_mask(contour, bbox):
    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    x_norm = (contour[:, 0] - float(x1)) / bbox_w
    y_norm = (contour[:, 1] - float(y1)) / bbox_h
    return np.logical_and.reduce(
        [
            x_norm >= BOTTOM_ARC_X_MIN,
            x_norm <= BOTTOM_ARC_X_MAX,
            y_norm >= BOTTOM_ARC_Y_MIN,
        ]
    )


def bottom_arc_support_score(mask, maps):
    bbox = mask_bbox(mask)
    contour = largest_contour(mask)
    if bbox is None or contour is None:
        return 0.0

    group = bottom_arc_mask(contour, bbox)
    if not np.any(group):
        return 0.0

    support = ensure_snap_support_map(maps, SNAP_DISTANCE_SIGMA)
    xs = contour[group, 0]
    ys = contour[group, 1]
    return float(np.mean(sample_float_map(support, xs, ys)))


def refine_bottom_arc(mask, maps):
    bbox = mask_bbox(mask)
    contour = largest_contour(mask)
    if bbox is None or contour is None:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "missing_contour"}

    group = bottom_arc_mask(contour, bbox)
    indices = np.flatnonzero(group)
    if indices.size < 16:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "group_too_small"}

    normals = outward_normals(mask, contour)
    support = ensure_snap_support_map(maps, SNAP_DISTANCE_SIGMA)

    pts = contour[indices]
    nrms = normals[indices]
    best_scores = np.full(indices.size, -1e9, dtype=np.float32)
    best_offsets = np.zeros(indices.size, dtype=np.float32)

    for off in BOTTOM_ARC_OFFSETS:
        xs = np.clip(pts[:, 0] + nrms[:, 0] * off, 0, mask.shape[1] - 1)
        ys = np.clip(pts[:, 1] + nrms[:, 1] * off, 0, mask.shape[0] - 1)
        values = sample_float_map(support, xs, ys) + float(off) * BOTTOM_ARC_OUTWARD_PRIOR
        better = values > best_scores
        best_scores[better] = values[better]
        best_offsets[better] = float(off)

    base_scores = sample_float_map(support, pts[:, 0], pts[:, 1])
    accepted = (best_offsets > 0.0) & ((best_scores - base_scores) >= BOTTOM_ARC_MIN_OFFSET_GAIN)
    moved_ratio = float(np.mean(accepted)) if accepted.size else 0.0
    if moved_ratio < BOTTOM_ARC_MIN_MOVED_RATIO:
        return mask, {
            "bottom_arc_refine_applied": False,
            "bottom_arc_refine_reason": "too_few_supported_points",
            "bottom_arc_refine_moved_ratio": moved_ratio,
        }

    best_offsets = np.where(accepted, best_offsets, 0.0).astype(np.float32)
    best_offsets = circular_smooth(best_offsets, BOTTOM_ARC_SMOOTH_WINDOW)

    snapped = contour.copy()
    snapped[indices] = contour[indices] + normals[indices] * best_offsets[:, None]
    snapped[:, 0] = np.clip(snapped[:, 0], 0, mask.shape[1] - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, mask.shape[0] - 1)

    snapped_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(snapped_mask, [np.round(snapped).astype(np.int32)], 255)
    snapped_mask = snapped_mask > 0

    if not snapped_mask[mask.shape[0] // 2, mask.shape[1] // 2]:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "center_lost"}

    iou = mask_iou_bool(mask, snapped_mask)
    if iou < BOTTOM_ARC_IOU_MIN:
        return mask, {
            "bottom_arc_refine_applied": False,
            "bottom_arc_refine_reason": "iou_too_low",
            "bottom_arc_refine_iou": float(iou),
        }

    area_gain_ratio = (int(snapped_mask.sum()) - int(mask.sum())) / float(max(1, int(mask.sum())))
    if area_gain_ratio > BOTTOM_ARC_MAX_AREA_GAIN_RATIO or area_gain_ratio < -BOTTOM_ARC_MAX_AREA_LOSS_RATIO:
        return mask, {
            "bottom_arc_refine_applied": False,
            "bottom_arc_refine_reason": "area_change_too_large",
            "bottom_arc_refine_area_gain_ratio": float(area_gain_ratio),
        }

    score_before = bottom_arc_support_score(mask, maps)
    score_after = bottom_arc_support_score(snapped_mask, maps)
    score_gain = score_after - score_before
    if score_gain < BOTTOM_ARC_MIN_SUPPORT_GAIN:
        return mask, {
            "bottom_arc_refine_applied": False,
            "bottom_arc_refine_reason": "support_gain_too_small",
            "bottom_arc_refine_support_before": float(score_before),
            "bottom_arc_refine_support_after": float(score_after),
            "bottom_arc_refine_support_gain": float(score_gain),
        }

    nonzero_offsets = best_offsets[best_offsets > 0]
    return snapped_mask, {
        "bottom_arc_refine_applied": True,
        "bottom_arc_refine_reason": "ok",
        "bottom_arc_refine_iou": float(iou),
        "bottom_arc_refine_area_gain_ratio": float(area_gain_ratio),
        "bottom_arc_refine_support_before": float(score_before),
        "bottom_arc_refine_support_after": float(score_after),
        "bottom_arc_refine_support_gain": float(score_gain),
        "bottom_arc_refine_moved_ratio": moved_ratio,
        "bottom_arc_refine_mean_offset": float(nonzero_offsets.mean()) if nonzero_offsets.size else 0.0,
        "bottom_arc_refine_max_offset": float(nonzero_offsets.max()) if nonzero_offsets.size else 0.0,
    }


def tight_mid_mask(contour, bbox):
    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    x_norm = (contour[:, 0] - float(x1)) / bbox_w
    y_norm = (contour[:, 1] - float(y1)) / bbox_h
    return np.logical_and.reduce(
        [
            x_norm >= TIGHT_MID_X_MIN,
            x_norm <= TIGHT_MID_X_MAX,
            y_norm >= TIGHT_MID_Y_MIN,
        ]
    )


def tight_mid_support_score(mask, maps):
    bbox = mask_bbox(mask)
    contour = largest_contour(mask)
    if bbox is None or contour is None:
        return 0.0
    group = tight_mid_mask(contour, bbox)
    if not np.any(group):
        return 0.0
    support = ensure_snap_support_map(maps, SNAP_DISTANCE_SIGMA)
    xs = contour[group, 0]
    ys = contour[group, 1]
    return float(np.mean(sample_float_map(support, xs, ys)))


def refine_tight_mid(mask, maps):
    bbox = mask_bbox(mask)
    contour = largest_contour(mask)
    if bbox is None or contour is None:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "missing_contour"}

    group = tight_mid_mask(contour, bbox)
    indices = np.flatnonzero(group)
    if indices.size < 12:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "group_too_small"}

    normals = outward_normals(mask, contour)
    support = ensure_snap_support_map(maps, SNAP_DISTANCE_SIGMA)

    pts = contour[indices]
    nrms = normals[indices]
    base_scores = sample_float_map(support, pts[:, 0], pts[:, 1])
    best_scores = base_scores.copy()
    best_offsets = np.zeros(indices.size, dtype=np.float32)

    for off in TIGHT_MID_OFFSETS:
        xs = np.clip(pts[:, 0] + nrms[:, 0] * off, 0, mask.shape[1] - 1)
        ys = np.clip(pts[:, 1] + nrms[:, 1] * off, 0, mask.shape[0] - 1)
        values = sample_float_map(support, xs, ys) + float(off) * TIGHT_MID_OUTWARD_PRIOR
        better = values > best_scores
        best_scores[better] = values[better]
        best_offsets[better] = float(off)

    accepted = (best_offsets > 0.0) & ((best_scores - base_scores) >= TIGHT_MID_MIN_OFFSET_GAIN)
    moved_ratio = float(np.mean(accepted)) if accepted.size else 0.0
    if moved_ratio < TIGHT_MID_MIN_MOVED_RATIO:
        return mask, {
            "tight_mid_refine_applied": False,
            "tight_mid_refine_reason": "too_few_supported_points",
            "tight_mid_refine_moved_ratio": moved_ratio,
        }

    best_offsets = np.where(accepted, best_offsets, 0.0).astype(np.float32)
    best_offsets = circular_smooth(best_offsets, TIGHT_MID_SMOOTH_WINDOW)

    snapped = contour.copy()
    snapped[indices] = contour[indices] + normals[indices] * best_offsets[:, None]
    snapped[:, 0] = np.clip(snapped[:, 0], 0, mask.shape[1] - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, mask.shape[0] - 1)

    snapped_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(snapped_mask, [np.round(snapped).astype(np.int32)], 255)
    snapped_mask = snapped_mask > 0

    if not snapped_mask[mask.shape[0] // 2, mask.shape[1] // 2]:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "center_lost"}

    iou = mask_iou_bool(mask, snapped_mask)
    if iou < TIGHT_MID_IOU_MIN:
        return mask, {
            "tight_mid_refine_applied": False,
            "tight_mid_refine_reason": "iou_too_low",
            "tight_mid_refine_iou": float(iou),
        }

    area_gain_ratio = (int(snapped_mask.sum()) - int(mask.sum())) / float(max(1, int(mask.sum())))
    if area_gain_ratio > TIGHT_MID_MAX_AREA_GAIN_RATIO or area_gain_ratio < -TIGHT_MID_MAX_AREA_LOSS_RATIO:
        return mask, {
            "tight_mid_refine_applied": False,
            "tight_mid_refine_reason": "area_change_too_large",
            "tight_mid_refine_area_gain_ratio": float(area_gain_ratio),
        }

    score_before = tight_mid_support_score(mask, maps)
    score_after = tight_mid_support_score(snapped_mask, maps)
    score_gain = score_after - score_before
    if score_gain < TIGHT_MID_MIN_SUPPORT_GAIN:
        return mask, {
            "tight_mid_refine_applied": False,
            "tight_mid_refine_reason": "support_gain_too_small",
            "tight_mid_refine_support_before": float(score_before),
            "tight_mid_refine_support_after": float(score_after),
            "tight_mid_refine_support_gain": float(score_gain),
        }

    nonzero_offsets = best_offsets[best_offsets > 0]
    return snapped_mask, {
        "tight_mid_refine_applied": True,
        "tight_mid_refine_reason": "ok",
        "tight_mid_refine_iou": float(iou),
        "tight_mid_refine_area_gain_ratio": float(area_gain_ratio),
        "tight_mid_refine_support_before": float(score_before),
        "tight_mid_refine_support_after": float(score_after),
        "tight_mid_refine_support_gain": float(score_gain),
        "tight_mid_refine_moved_ratio": moved_ratio,
        "tight_mid_refine_mean_offset": float(nonzero_offsets.mean()) if nonzero_offsets.size else 0.0,
        "tight_mid_refine_max_offset": float(nonzero_offsets.max()) if nonzero_offsets.size else 0.0,
    }


def _update_fit_meta(meta, mask, maps, sigma):
    final_stats = compute_mask_border_stats(mask)
    final_fit = measure_mask_fit(mask, maps, sigma)
    if final_stats is not None:
        meta["coverage"] = float(mask.mean())
        meta["left_gap"] = int(final_stats["left_gap"])
        meta["bottom_gap"] = int(final_stats["bottom_gap"])
        meta["left_touch"] = float(final_stats["left_touch"])
        meta["bottom_touch"] = float(final_stats["bottom_touch"])
    if final_fit is not None:
        meta["boundary_response"] = float(final_fit["boundary_response"])
        meta["distance_score"] = float(final_fit["distance_score"])
        meta["contrast_inside"] = float(final_fit["contrast_inside"])
        meta["contrast_outside"] = float(final_fit["contrast_outside"])
        meta["top_score"] = float(final_fit["top_score"])
        meta["right_score"] = float(final_fit["right_score"])
        meta["bottom_score"] = float(final_fit["bottom_score"])
        meta["left_score"] = float(final_fit["left_score"])
        meta["bottom_mid_score"] = float(final_fit["bottom_mid_score"])
        meta["side_score"] = float(final_fit["side_score"])
    return meta


def find_best_template_fit(background_bgr, template_mask):
    full_shape = background_bgr.shape[:2]
    scaled_shape = (
        max(64, int(round(full_shape[0] * SEARCH_IMAGE_SCALE))),
        max(64, int(round(full_shape[1] * SEARCH_IMAGE_SCALE))),
    )

    background_small = cv2.resize(background_bgr, (scaled_shape[1], scaled_shape[0]), interpolation=cv2.INTER_AREA)
    template_small = resize_mask_to_shape(template_mask, scaled_shape)
    template_small = template_small.astype(bool)

    maps_small = build_response_maps(background_small)
    maps_small["scale_factor"] = SEARCH_IMAGE_SCALE
    maps_full = build_response_maps(background_bgr)
    maps_full["scale_factor"] = 1.0

    template_coverage = float(template_mask.mean())
    print("  template fit: coarse search on sampled background")
    coarse_best = search_stage(
        template_small,
        template_coverage,
        maps_small,
        dx_values=build_search_range(0, SEARCH_DX_RANGE, COARSE_DX_STEP),
        dy_values=build_search_range(0, SEARCH_DY_RANGE, COARSE_DY_STEP),
        sx_values=list(COARSE_SCALE_VALUES),
        sy_values=list(COARSE_SCALE_VALUES),
        dx_range_full=SEARCH_DX_RANGE,
        dy_range_full=SEARCH_DY_RANGE,
        sigma=DISTANCE_SIGMA_SMALL,
    )

    if coarse_best is None:
        return None, maps_full, maps_small, {"reason": "no_valid_candidate"}

    print("  template fit: local refinement")
    refine_best = search_stage(
        template_small,
        template_coverage,
        maps_small,
        dx_values=build_search_range(coarse_best["dx_full"], REFINE_DX_RADIUS, REFINE_DX_STEP),
        dy_values=build_search_range(coarse_best["dy_full"], REFINE_DY_RADIUS, REFINE_DY_STEP),
        sx_values=iter_values_around(coarse_best["sx"], REFINE_SCALE_OFFSETS),
        sy_values=iter_values_around(coarse_best["sy"], REFINE_SCALE_OFFSETS),
        dx_range_full=SEARCH_DX_RANGE,
        dy_range_full=SEARCH_DY_RANGE,
        sigma=DISTANCE_SIGMA_SMALL,
    )

    if refine_best is None:
        refine_best = coarse_best

    print("  template fit: full-resolution refinement")
    final_best = search_stage(
        template_mask,
        template_coverage,
        maps_full,
        dx_values=build_search_range(refine_best["dx_full"], FINAL_DX_RADIUS, FINAL_DX_STEP),
        dy_values=build_search_range(refine_best["dy_full"], FINAL_DY_RADIUS, FINAL_DY_STEP),
        sx_values=iter_values_around(refine_best["sx"], FINAL_SCALE_OFFSETS),
        sy_values=iter_values_around(refine_best["sy"], FINAL_SCALE_OFFSETS),
        dx_range_full=SEARCH_DX_RANGE,
        dy_range_full=SEARCH_DY_RANGE,
        sigma=DISTANCE_SIGMA_FULL,
    )

    if final_best is None:
        final_best = refine_best

    print("  template fit: local left-bottom patch")
    patched_mask, patch_meta = apply_local_left_bottom_patch(final_best["mask"])
    final_best["mask"] = patched_mask
    patched_stats = compute_mask_border_stats(patched_mask)
    if patched_stats is not None:
        final_best["coverage"] = float(patched_mask.mean())
        final_best["left_gap"] = int(patched_stats["left_gap"])
        final_best["right_gap"] = int(patched_stats["right_gap"])
        final_best["top_gap"] = int(patched_stats["top_gap"])
        final_best["bottom_gap"] = int(patched_stats["bottom_gap"])
        final_best["left_touch"] = float(patched_stats["left_touch"])
        final_best["bottom_touch"] = float(patched_stats["bottom_touch"])

    meta = {
        "reason": "ok",
        "score": float(final_best["score"]),
        "coverage": float(final_best["coverage"]),
        "boundary_response": float(final_best["boundary_response"]),
        "distance_score": float(final_best["distance_score"]),
        "contrast_inside": float(final_best["contrast_inside"]),
        "contrast_outside": float(final_best["contrast_outside"]),
        "dx_full": int(final_best["dx_full"]),
        "dy_full": int(final_best["dy_full"]),
        "sx": float(final_best["sx"]),
        "sy": float(final_best["sy"]),
        "left_gap": int(final_best["left_gap"]),
        "bottom_gap": int(final_best["bottom_gap"]),
        "left_touch": float(final_best["left_touch"]),
        "bottom_touch": float(final_best["bottom_touch"]),
    }
    meta.update(patch_meta)

    print("  template fit: black-edge snap")
    snapped_mask, snap_meta = snap_mask_to_black_edge(final_best["mask"], maps_full)
    meta = _update_fit_meta(meta, snapped_mask, maps_full, SNAP_DISTANCE_SIGMA)
    meta.update(snap_meta)

    print("  template fit: bottom-arc refine")
    bottom_mask, bottom_meta = refine_bottom_arc(snapped_mask, maps_full)
    meta = _update_fit_meta(meta, bottom_mask, maps_full, SNAP_DISTANCE_SIGMA)
    meta.update(bottom_meta)

    print("  template fit: tight-mid refine")
    tight_mask, tight_meta = refine_tight_mid(bottom_mask, maps_full)
    meta = _update_fit_meta(meta, tight_mask, maps_full, SNAP_DISTANCE_SIGMA)
    meta.update(tight_meta)

    print("  template fit: final black-edge snap")
    final_mask, final_snap_meta = snap_mask_to_black_edge(tight_mask, maps_full)
    meta = _update_fit_meta(meta, final_mask, maps_full, SNAP_DISTANCE_SIGMA)
    meta.update(final_snap_meta)

    return final_mask, maps_full, maps_small, meta


def build_folder_shared_chamber_assets(folder_path):
    images = collect_folder_images(folder_path)
    if not images:
        return None

    sampled_images = sample_images_uniformly(images, BACKGROUND_SAMPLE_COUNT)
    print(f"  chamber sampling: using {len(sampled_images)}/{len(images)} image(s) from {folder_path}")

    background_bgr = build_sampled_background(sampled_images)
    if background_bgr is None:
        return None

    template_mask = load_reference_template_mask(background_bgr.shape[:2])
    if template_mask is None:
        return {
            "images": images,
            "sampled_images": sampled_images,
            "background_bgr": background_bgr,
            "chamber_mask": None,
            "chamber_core_mask": None,
            "template_mask": None,
            "maps_full": None,
            "maps_small": None,
            "meta": {"reason": "missing_template"},
        }

    chamber_mask, maps_full, maps_small, meta = find_best_template_fit(background_bgr, template_mask)
    chamber_core_mask = build_chamber_core_mask(chamber_mask) if chamber_mask is not None else None

    if chamber_mask is not None and ENABLE_CHAMBER_FILTER and CHAMBER_MASK_DILATION > 0:
        kernel_size = ensure_odd(CHAMBER_MASK_DILATION)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        chamber_mask = cv2.dilate(chamber_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        chamber_core_mask = build_chamber_core_mask(chamber_mask)

    return {
        "images": images,
        "sampled_images": sampled_images,
        "background_bgr": background_bgr,
        "chamber_mask": chamber_mask,
        "chamber_core_mask": chamber_core_mask,
        "template_mask": template_mask,
        "maps_full": maps_full,
        "maps_small": maps_small,
        "meta": meta,
    }


def save_shared_debug_artifacts(folder_path, shared_assets):
    if shared_assets is None:
        return

    results_dir = os.path.join(folder_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    background_bgr = shared_assets["background_bgr"]
    chamber_mask = shared_assets.get("chamber_mask")
    template_mask = shared_assets.get("template_mask")
    maps_full = shared_assets.get("maps_full")
    meta = shared_assets.get("meta") or {}

    background_path = os.path.join(results_dir, "shared_bgfit_background.png")
    response_path = os.path.join(results_dir, "shared_bgfit_response.png")
    edge_path = os.path.join(results_dir, "shared_bgfit_edge.png")
    chamber_path = os.path.join(results_dir, "shared_bgfit_mask.png")
    overlay_path = os.path.join(results_dir, "shared_bgfit_overlay.png")
    template_overlay_path = os.path.join(results_dir, "shared_bgfit_template_overlay.png")
    samples_txt_path = os.path.join(results_dir, "shared_bgfit_samples.txt")
    meta_txt_path = os.path.join(results_dir, "shared_bgfit_meta.txt")

    overlay = background_bgr.copy()
    if chamber_mask is not None:
        overlay = overlay_chamber_boundary(overlay, chamber_mask)

    template_overlay = background_bgr.copy()
    if template_mask is not None:
        template_overlay = overlay_chamber_boundary(template_overlay, template_mask)
    if chamber_mask is not None:
        template_overlay = overlay_chamber_boundary(
            template_overlay,
            chamber_mask,
            alpha=0.60,
            boundary_color=(0, 255, 255),
        )

    cv2.imwrite(background_path, background_bgr)
    if maps_full is not None:
        response_u8 = np.clip(maps_full["response"] * 255.0, 0, 255).astype(np.uint8)
        edge_u8 = maps_full["edge"].astype(np.uint8) * 255
        cv2.imwrite(response_path, response_u8)
        cv2.imwrite(edge_path, edge_u8)
    if chamber_mask is not None:
        cv2.imwrite(chamber_path, chamber_mask.astype(np.uint8) * 255)
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(template_overlay_path, template_overlay)

    with open(samples_txt_path, "w", encoding="utf-8") as f:
        for image_path in shared_assets["sampled_images"]:
            f.write(f"{os.path.basename(image_path)}\n")

    with open(meta_txt_path, "w", encoding="utf-8") as f:
        for key in sorted(meta.keys()):
            f.write(f"{key}: {meta[key]}\n")


def process_image_with_shared_mask(model, image_path, chamber_mask, chamber_core_mask):
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

    if ENABLE_CHAMBER_FILTER and chamber_mask is not None:
        base_instances = filter_instances_inside_chamber(base_instances, chamber_mask, chamber_core_mask)
        tile_instances = filter_instances_inside_chamber(tile_instances, chamber_mask, chamber_core_mask)

    merged_instances = merge_instances(
        base_instances + tile_instances,
        iou_threshold=MERGE_IOU_THRESHOLD,
        class_agnostic=True,
    )

    if ENABLE_CHAMBER_FILTER and chamber_mask is not None:
        merged_instances = filter_instances_inside_chamber(merged_instances, chamber_mask, chamber_core_mask)

    if ENABLE_MASK_REFINEMENT:
        merged_instances = refine_instances(merged_instances)

    if ENABLE_TINY_INSTANCE_FILTER:
        merged_instances = filter_tiny_instances(merged_instances)

    if not merged_instances:
        raise RuntimeError("no detections from full pass or tile pass")

    return {
        "original_img": original_img,
        "merged_instances": merged_instances,
        "full_count": len(base_instances),
        "tile_count": len(tile_instances),
        "cfg_used": cfg_used,
        "chamber_mask": chamber_mask if ENABLE_CHAMBER_FILTER else None,
    }


def process_image(model, image_path):
    global CURRENT_SHARED_CHAMBER_MASK
    global CURRENT_SHARED_CHAMBER_CORE_MASK

    if CURRENT_SHARED_CHAMBER_MASK is None:
        return ORIGINAL_PROCESS_IMAGE(model, image_path)

    return process_image_with_shared_mask(
        model,
        image_path,
        CURRENT_SHARED_CHAMBER_MASK,
        CURRENT_SHARED_CHAMBER_CORE_MASK,
    )


def process_folder(folder_path, model):
    global CURRENT_SHARED_CHAMBER_MASK
    global CURRENT_SHARED_CHAMBER_CORE_MASK
    global CURRENT_SHARED_DEBUG

    shared_assets = build_folder_shared_chamber_assets(folder_path)
    meta = (shared_assets or {}).get("meta") or {}
    shared_score = float(meta.get("score", -1.0))

    if shared_assets is not None and shared_assets.get("chamber_mask") is not None and shared_score >= MIN_SHARED_SCORE:
        CURRENT_SHARED_CHAMBER_MASK = shared_assets["chamber_mask"]
        CURRENT_SHARED_CHAMBER_CORE_MASK = shared_assets["chamber_core_mask"]
        CURRENT_SHARED_DEBUG = shared_assets
        print(
            f"  shared chamber ready: "
            f"score={shared_score:.3f}, "
            f"dx={meta.get('dx_full', 0)}, dy={meta.get('dy_full', 0)}, "
            f"sx={meta.get('sx', 1.0):.3f}, sy={meta.get('sy', 1.0):.3f}, "
            f"coverage={meta.get('coverage', -1):.3f}, "
            f"left_gap={meta.get('left_gap', -1)}, bottom_gap={meta.get('bottom_gap', -1)}"
        )
    else:
        CURRENT_SHARED_CHAMBER_MASK = None
        CURRENT_SHARED_CHAMBER_CORE_MASK = None
        CURRENT_SHARED_DEBUG = shared_assets
        print(
            "  warning: failed to build shared chamber mask, "
            f"reason={meta.get('reason', 'score_too_low')}, "
            f"score={shared_score:.3f}; falling back to per-image chamber detection"
        )

    try:
        ORIGINAL_PROCESS_FOLDER(folder_path, model)
        if CURRENT_SHARED_DEBUG is not None:
            save_shared_debug_artifacts(folder_path, CURRENT_SHARED_DEBUG)
    finally:
        CURRENT_SHARED_CHAMBER_MASK = None
        CURRENT_SHARED_CHAMBER_CORE_MASK = None
        CURRENT_SHARED_DEBUG = None


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
