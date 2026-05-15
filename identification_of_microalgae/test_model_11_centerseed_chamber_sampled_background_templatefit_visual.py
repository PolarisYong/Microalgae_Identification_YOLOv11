from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np


BASE_FILE = Path(__file__).with_name("test_model_11_centerseed_chamber_borderaware.py")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"base script not found: {BASE_FILE}")

spec = importlib.util.spec_from_file_location("microalgae_sampled_background_templatefit_base", BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load base script spec: {BASE_FILE}")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


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
LOCAL_PATCH_MAX_AREA_GAIN_RATIO = 0.12
LOCAL_PATCH_MIN_ROI_SIZE = 140
LOCAL_PATCH_MAX_ROI_SIZE = 320

TEMPLATE_NAMES = (
    "chamber_template.png",
    "chamber_mask.png",
    "chamber_template2.png",
    "chamber_mask2.png",
)

TEMPLATE_SEARCH_DIRS = (
    Path(r"E:\pythonProject\Microalgae_Identification_YOLOv11\identification_of_microalgae"),
)


CURRENT_SHARED_CHAMBER_MASK = None
CURRENT_SHARED_CHAMBER_CORE_MASK = None
CURRENT_SHARED_DEBUG = None
ORIGINAL_PROCESS_IMAGE = base.process_image
ORIGINAL_PROCESS_FOLDER = base.process_folder


def collect_folder_images(folder_path):
    image_extensions = [".tif", ".jpg", ".jpeg", ".png"]
    images = set()
    for ext in image_extensions:
        images.update(glob.glob(os.path.join(folder_path, f"*{ext}")))
        images.update(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(images, key=base.extract_number)


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
        image_bgr = base.read_image_bgr(image_path)
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
        *TEMPLATE_SEARCH_DIRS,
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]

    for base_dir in search_dirs:
        for name in TEMPLATE_NAMES:
            mask_path = base_dir / name
            if not mask_path.exists():
                continue
            try:
                mask_img = base.Image.open(str(mask_path)).convert("L")
                mask = np.asarray(mask_img)
                if mask.shape[:2] != reference_shape:
                    mask = cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_NEAREST)
                print(f"  reference template loaded: {mask_path}")
                return mask > 127
            except Exception as exc:
                print(f"  warning: failed to load template {mask_path}: {type(exc).__name__}: {exc}")

    return None


def ensure_odd(value):
    value = max(1, int(round(value)))
    if value % 2 == 0:
        value += 1
    return value


def resize_bool_mask(mask, shape_hw):
    if mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


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
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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
    if left_gap > LOCAL_PATCH_MAX_LEFT_GAP or bottom_gap > LOCAL_PATCH_MAX_BOTTOM_GAP:
        return mask, {"patch_applied": False, "patch_reason": "gap_too_large"}

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


def find_best_template_fit(background_bgr, template_mask):
    full_shape = background_bgr.shape[:2]
    scaled_shape = (
        max(64, int(round(full_shape[0] * SEARCH_IMAGE_SCALE))),
        max(64, int(round(full_shape[1] * SEARCH_IMAGE_SCALE))),
    )

    background_small = cv2.resize(background_bgr, (scaled_shape[1], scaled_shape[0]), interpolation=cv2.INTER_AREA)
    template_small = resize_bool_mask(template_mask, scaled_shape)

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
    return final_best["mask"], maps_full, maps_small, meta


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
            "meta": {"reason": "missing_template"},
        }

    chamber_mask, maps_full, maps_small, meta = find_best_template_fit(background_bgr, template_mask)
    chamber_core_mask = base.build_chamber_core_mask(chamber_mask) if chamber_mask is not None else None

    if chamber_mask is not None and base.ENABLE_CHAMBER_FILTER and base.CHAMBER_MASK_DILATION > 0:
        kernel_size = ensure_odd(base.CHAMBER_MASK_DILATION)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        chamber_mask = cv2.dilate(chamber_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0
        chamber_core_mask = base.build_chamber_core_mask(chamber_mask)

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
        overlay = base.overlay_chamber_boundary(overlay, chamber_mask)

    template_overlay = background_bgr.copy()
    if template_mask is not None:
        template_overlay = base.overlay_chamber_boundary(template_overlay, template_mask)
    if chamber_mask is not None:
        template_overlay = base.overlay_chamber_boundary(
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
    original_img = base.read_image_bgr(image_path)
    if original_img is None:
        raise ValueError("unable to read image")

    working_img = base.preprocess_for_dense(original_img) if base.ENABLE_DENSE_PREPROCESS else original_img

    full_results, cfg_used, pred_error, full_had_oom = base.predict_with_retry(model, working_img, mode="full")
    base_instances = []
    if full_results is not None and len(full_results) > 0:
        full_det = full_results[0]
        base_instances = base.extract_instances_from_result(full_det, offset_x=0, offset_y=0)
    else:
        print(f"  full-image inference returned no detections: {pred_error}")

    tile_instances = []
    tile_had_oom = False
    force_cpu = full_had_oom or base.is_oom_text(pred_error)

    if base.ENABLE_TILE_PASS:
        tile_instances, tile_had_oom = base.predict_tiles(model, working_img, cpu_only=force_cpu)

    if not tile_instances and (force_cpu or tile_had_oom):
        print("  switching to CPU fallback tile pass")
        cpu_tile_instances, _ = base.predict_tiles(model, working_img, cpu_only=True)
        tile_instances.extend(cpu_tile_instances)

    if base.ENABLE_CHAMBER_FILTER and chamber_mask is not None:
        base_instances = base.filter_instances_inside_chamber(base_instances, chamber_mask, chamber_core_mask)
        tile_instances = base.filter_instances_inside_chamber(tile_instances, chamber_mask, chamber_core_mask)

    merged_instances = base.merge_instances(
        base_instances + tile_instances,
        iou_threshold=base.MERGE_IOU_THRESHOLD,
        class_agnostic=True,
    )

    if base.ENABLE_CHAMBER_FILTER and chamber_mask is not None:
        merged_instances = base.filter_instances_inside_chamber(merged_instances, chamber_mask, chamber_core_mask)

    if base.ENABLE_MASK_REFINEMENT:
        merged_instances = base.refine_instances(merged_instances)

    if base.ENABLE_TINY_INSTANCE_FILTER:
        merged_instances = base.filter_tiny_instances(merged_instances)

    if not merged_instances:
        raise RuntimeError("no detections from full pass or tile pass")

    return {
        "original_img": original_img,
        "merged_instances": merged_instances,
        "full_count": len(base_instances),
        "tile_count": len(tile_instances),
        "cfg_used": cfg_used,
        "chamber_mask": chamber_mask if base.ENABLE_CHAMBER_FILTER else None,
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


base.process_image = process_image
base.process_folder = process_folder


def main():
    base.main()


if __name__ == "__main__":
    main()
