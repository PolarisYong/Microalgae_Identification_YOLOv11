from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


BASE_FILE = Path(__file__).with_name("test_model_11_centerseed_chamber_sampled_background_templatefit_visual.py")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"base script not found: {BASE_FILE}")

spec = importlib.util.spec_from_file_location("microalgae_templatefit_blackedgesnap_base", BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load base script spec: {BASE_FILE}")

basefit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basefit)
ORIGINAL_FIND_BEST_TEMPLATE_FIT = basefit.find_best_template_fit


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


def mask_iou(mask_a, mask_b):
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

    dark_norm = basefit.robust_normalize(255.0 - maps["gray"], low_q=55.0, high_q=99.5)
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
    bbox = basefit.mask_bbox(mask)
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
    boundary = basefit.build_boundary_band(mask)
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

    inner_core = basefit.build_inner_core(mask)
    outer_ring = basefit.build_outer_ring(mask)
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

    solve(np.logical_and(groups["left_bottom"], remaining), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], 0.010, 0.070, outward_bonus=0.0018)
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
    bbox = basefit.mask_bbox(mask)
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

    iou = mask_iou(mask, snapped_mask)
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


def find_best_template_fit(background_bgr, template_mask):
    chamber_mask, maps_full, maps_small, meta = ORIGINAL_FIND_BEST_TEMPLATE_FIT(background_bgr, template_mask)
    if chamber_mask is None:
        return chamber_mask, maps_full, maps_small, meta

    print("  template fit: black-edge snap")
    snapped_mask, snap_meta = snap_mask_to_black_edge(chamber_mask, maps_full)
    final_stats = basefit.compute_mask_border_stats(snapped_mask)
    final_fit = measure_mask_fit(snapped_mask, maps_full, SNAP_DISTANCE_SIGMA)
    if final_stats is not None:
        meta["coverage"] = float(snapped_mask.mean())
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
    meta.update(snap_meta)
    return snapped_mask, maps_full, maps_small, meta


basefit.find_best_template_fit = find_best_template_fit


def main():
    basefit.main()


if __name__ == "__main__":
    main()
