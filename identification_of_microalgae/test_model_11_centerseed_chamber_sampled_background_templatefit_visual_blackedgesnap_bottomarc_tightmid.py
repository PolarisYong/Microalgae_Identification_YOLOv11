from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


BASE_FILE = Path(__file__).with_name("test_model_11_centerseed_chamber_sampled_background_templatefit_visual_blackedgesnap_bottomarc.py")
if not BASE_FILE.exists():
    raise FileNotFoundError(f"base script not found: {BASE_FILE}")

spec = importlib.util.spec_from_file_location("microalgae_templatefit_blackedgesnap_bottomarc_tightmid_base", BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load base script spec: {BASE_FILE}")

bottomarc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bottomarc)
ORIGINAL_FIND_BEST_TEMPLATE_FIT = bottomarc.find_best_template_fit


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
    bbox = bottomarc.blackedge.basefit.mask_bbox(mask)
    contour = bottomarc.blackedge.largest_contour(mask)
    if bbox is None or contour is None:
        return 0.0
    group = tight_mid_mask(contour, bbox)
    if not np.any(group):
        return 0.0
    support = bottomarc.blackedge.ensure_snap_support_map(maps, bottomarc.blackedge.SNAP_DISTANCE_SIGMA)
    xs = contour[group, 0]
    ys = contour[group, 1]
    return float(np.mean(bottomarc.blackedge.sample_float_map(support, xs, ys)))


def refine_tight_mid(mask, maps):
    bbox = bottomarc.blackedge.basefit.mask_bbox(mask)
    contour = bottomarc.blackedge.largest_contour(mask)
    if bbox is None or contour is None:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "missing_contour"}

    group = tight_mid_mask(contour, bbox)
    indices = np.flatnonzero(group)
    if indices.size < 12:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "group_too_small"}

    normals = bottomarc.blackedge.outward_normals(mask, contour)
    support = bottomarc.blackedge.ensure_snap_support_map(maps, bottomarc.blackedge.SNAP_DISTANCE_SIGMA)

    pts = contour[indices]
    nrms = normals[indices]
    base_scores = bottomarc.blackedge.sample_float_map(support, pts[:, 0], pts[:, 1])
    best_scores = base_scores.copy()
    best_offsets = np.zeros(indices.size, dtype=np.float32)

    for off in TIGHT_MID_OFFSETS:
        xs = np.clip(pts[:, 0] + nrms[:, 0] * off, 0, mask.shape[1] - 1)
        ys = np.clip(pts[:, 1] + nrms[:, 1] * off, 0, mask.shape[0] - 1)
        values = bottomarc.blackedge.sample_float_map(support, xs, ys) + float(off) * TIGHT_MID_OUTWARD_PRIOR
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
    best_offsets = bottomarc.blackedge.circular_smooth(best_offsets, TIGHT_MID_SMOOTH_WINDOW)

    snapped = contour.copy()
    snapped[indices] = contour[indices] + normals[indices] * best_offsets[:, None]
    snapped[:, 0] = np.clip(snapped[:, 0], 0, mask.shape[1] - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, mask.shape[0] - 1)

    snapped_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(snapped_mask, [np.round(snapped).astype(np.int32)], 255)
    snapped_mask = snapped_mask > 0

    if not snapped_mask[mask.shape[0] // 2, mask.shape[1] // 2]:
        return mask, {"tight_mid_refine_applied": False, "tight_mid_refine_reason": "center_lost"}

    iou = bottomarc.blackedge.mask_iou(mask, snapped_mask)
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


def find_best_template_fit(background_bgr, template_mask):
    chamber_mask, maps_full, maps_small, meta = ORIGINAL_FIND_BEST_TEMPLATE_FIT(background_bgr, template_mask)
    if chamber_mask is None:
        return chamber_mask, maps_full, maps_small, meta

    print("  template fit: tight-mid refine")
    refined_mask, refine_meta = refine_tight_mid(chamber_mask, maps_full)

    final_stats = bottomarc.blackedge.basefit.compute_mask_border_stats(refined_mask)
    final_fit = bottomarc.blackedge.measure_mask_fit(refined_mask, maps_full, bottomarc.blackedge.SNAP_DISTANCE_SIGMA)
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


bottomarc.blackedge.basefit.find_best_template_fit = find_best_template_fit


def main():
    bottomarc.blackedge.basefit.main()


if __name__ == "__main__":
    main()
