from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


BASE_FILE = Path(__file__).with_name("test_model_11_centerseed_chamber_sampled_background_templatefit_visual_blackedgesnap.py")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"base script not found: {BASE_FILE}")

spec = importlib.util.spec_from_file_location("microalgae_templatefit_blackedgesnap_bottomarc_base", BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load base script spec: {BASE_FILE}")

blackedge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blackedge)
ORIGINAL_FIND_BEST_TEMPLATE_FIT = blackedge.find_best_template_fit


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
    bbox = blackedge.basefit.mask_bbox(mask)
    contour = blackedge.largest_contour(mask)
    if bbox is None or contour is None:
        return 0.0

    group = bottom_arc_mask(contour, bbox)
    if not np.any(group):
        return 0.0

    support = blackedge.ensure_snap_support_map(maps, blackedge.SNAP_DISTANCE_SIGMA)
    xs = contour[group, 0]
    ys = contour[group, 1]
    return float(np.mean(blackedge.sample_float_map(support, xs, ys)))


def refine_bottom_arc(mask, maps):
    bbox = blackedge.basefit.mask_bbox(mask)
    contour = blackedge.largest_contour(mask)
    if bbox is None or contour is None:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "missing_contour"}

    group = bottom_arc_mask(contour, bbox)
    indices = np.flatnonzero(group)
    if indices.size < 16:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "group_too_small"}

    normals = blackedge.outward_normals(mask, contour)
    support = blackedge.ensure_snap_support_map(maps, blackedge.SNAP_DISTANCE_SIGMA)

    pts = contour[indices]
    nrms = normals[indices]
    best_scores = np.full(indices.size, -1e9, dtype=np.float32)
    best_offsets = np.zeros(indices.size, dtype=np.float32)

    for off in BOTTOM_ARC_OFFSETS:
        xs = np.clip(pts[:, 0] + nrms[:, 0] * off, 0, mask.shape[1] - 1)
        ys = np.clip(pts[:, 1] + nrms[:, 1] * off, 0, mask.shape[0] - 1)
        values = blackedge.sample_float_map(support, xs, ys) + float(off) * BOTTOM_ARC_OUTWARD_PRIOR
        better = values > best_scores
        best_scores[better] = values[better]
        best_offsets[better] = float(off)

    base_scores = blackedge.sample_float_map(support, pts[:, 0], pts[:, 1])
    accepted = (best_offsets > 0.0) & ((best_scores - base_scores) >= BOTTOM_ARC_MIN_OFFSET_GAIN)
    moved_ratio = float(np.mean(accepted)) if accepted.size else 0.0
    if moved_ratio < BOTTOM_ARC_MIN_MOVED_RATIO:
        return mask, {
            "bottom_arc_refine_applied": False,
            "bottom_arc_refine_reason": "too_few_supported_points",
            "bottom_arc_refine_moved_ratio": moved_ratio,
        }

    best_offsets = np.where(accepted, best_offsets, 0.0).astype(np.float32)
    best_offsets = blackedge.circular_smooth(best_offsets, BOTTOM_ARC_SMOOTH_WINDOW)

    snapped = contour.copy()
    snapped[indices] = contour[indices] + normals[indices] * best_offsets[:, None]
    snapped[:, 0] = np.clip(snapped[:, 0], 0, mask.shape[1] - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, mask.shape[0] - 1)

    snapped_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(snapped_mask, [np.round(snapped).astype(np.int32)], 255)
    snapped_mask = snapped_mask > 0

    if not snapped_mask[mask.shape[0] // 2, mask.shape[1] // 2]:
        return mask, {"bottom_arc_refine_applied": False, "bottom_arc_refine_reason": "center_lost"}

    iou = blackedge.mask_iou(mask, snapped_mask)
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


def find_best_template_fit(background_bgr, template_mask):
    chamber_mask, maps_full, maps_small, meta = ORIGINAL_FIND_BEST_TEMPLATE_FIT(background_bgr, template_mask)
    if chamber_mask is None:
        return chamber_mask, maps_full, maps_small, meta

    print("  template fit: bottom-arc refine")
    refined_mask, refine_meta = refine_bottom_arc(chamber_mask, maps_full)

    final_stats = blackedge.basefit.compute_mask_border_stats(refined_mask)
    final_fit = blackedge.measure_mask_fit(refined_mask, maps_full, blackedge.SNAP_DISTANCE_SIGMA)
    if final_stats is not None:
        meta["coverage"] = float(refined_mask.mean())
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
    meta.update(refine_meta)
    return refined_mask, maps_full, maps_small, meta


blackedge.basefit.find_best_template_fit = find_best_template_fit


def main():
    blackedge.basefit.main()


if __name__ == "__main__":
    main()
