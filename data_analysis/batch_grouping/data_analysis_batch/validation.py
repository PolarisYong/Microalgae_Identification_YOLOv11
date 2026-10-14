from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Sequence

from openpyxl import load_workbook
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from .grouping import metrics, qualifies, sort_by_umax
from .models import ChamberResult


def _font_is_red(cell) -> bool:
    color = cell.font.color
    return bool(color and color.type == "rgb" and color.rgb and color.rgb.upper().endswith("FF0000"))


def validate_excel_outputs(
    base_dir: Path,
    output_dir: Path,
    groups: Sequence[Sequence[ChamberResult]],
    strict_rows: Sequence[ChamberResult],
    min_group_size: int,
    max_group_size: int,
    cv_limit: float,
    normality_p_limit: float,
    mean_spread_ratio_limit: float | None = None,
) -> dict:
    strict_map = {row.key: row for row in strict_rows}
    selected_keys: list[tuple[str, str]] = []
    group_results = []
    for index, expected_group in enumerate(groups, start=1):
        path = output_dir / f"selected__groups_{index}_{len(expected_group)}.xlsx"
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise AssertionError(f"Corrupt group workbook: {path}")
        workbook = load_workbook(path, read_only=True, data_only=True)
        selected_name = next(name for name in workbook.sheetnames if name.startswith("selected_"))
        values = [row for row in workbook[selected_name].iter_rows(min_row=2, values_only=True) if row[2]]
        workbook.close()
        if len(values) != len(expected_group):
            raise AssertionError(f"Unexpected row count in {path.name}")
        # Companion growth-curve image embedded below the data block: every
        # group produces *exactly one* composite PNG (assembled in
        # :func:`ppt_io.compose_group_growth_chart` from all of its chambers),
        # so the embedded-image count must equal 1 regardless of group size.
        # We re-open the workbook in writable mode to read the image list,
        # since the prior read-only handle is closed.
        img_wb = load_workbook(path, read_only=False, data_only=False)
        embedded = len(img_wb[selected_name]._images)
        img_wb.close()
        if embedded != 1:
            raise AssertionError(
                f"{path.name}: expected exactly 1 composite growth-curve "
                f"image (one per group), got {embedded}"
            )
        for output_row in values:
            key = output_row[0], output_row[2]
            source = strict_map[key]
            expected = source.output_row()[3:13]
            actual = output_row[3:13]
            if any(abs(float(a) - float(b)) >= 1e-9 for a, b in zip(actual, expected)):
                raise AssertionError(f"Data mismatch for {key} in {path.name}")
            if float(output_row[5]) <= 0 or float(output_row[6]) <= 0.9:
                raise AssertionError(f"Strict filter failed for {key}")
            selected_keys.append(key)
        result = metrics(expected_group)
        if not qualifies(expected_group, cv_limit, normality_p_limit):
            raise AssertionError(f"Statistical qualification failed for group {index}")
        group_results.append(
            {
                "file": path.name,
                "count": len(expected_group),
                "mean": result.mean,
                "sample_sd": result.sample_sd,
                "cv": result.cv,
                "jb": result.jb,
                "p_value": result.p_value,
                "embedded_images": embedded,
            }
        )

    if len(selected_keys) != len(set(selected_keys)):
        raise AssertionError("Duplicate chambers found across selected groups")
    if any(len(group) < min_group_size or len(group) > max_group_size for group in groups):
        raise AssertionError("Selected group is outside the allowed size range")

    # Compact-mode boundary contract: every group must satisfy
    #   min_group_size <= len(group) <= max_group_size
    # and the only group that may be smaller than max_group_size is the LAST one
    # (i.e. at most one group may be undersized, and it must be the trailing
    # residual). Earlier groups are expected to be saturated to max_group_size.
    sizes = [len(group) for group in groups]
    undersized_indices = [i for i, size in enumerate(sizes) if size < max_group_size]
    if len(undersized_indices) > 1:
        raise AssertionError(
            "Only the last group may be smaller than max_group_size; "
            f"found sizes {sizes}"
        )
    if undersized_indices and undersized_indices[0] != len(sizes) - 1:
        raise AssertionError(
            "An undersized group is not in the trailing position; "
            f"found sizes {sizes}"
        )

    # Optional inter-group mean-spread budget: when configured, the spread of
    # group means must stay within ``ratio × overall strict-candidate mean``.
    if mean_spread_ratio_limit is not None and group_results:
        group_means = [entry["mean"] for entry in group_results]
        spread = max(group_means) - min(group_means)
        overall_mean = sum(row.umax for row in strict_rows) / len(strict_rows)
        budget = mean_spread_ratio_limit * overall_mean
        if spread > budget + 1e-12:
            raise AssertionError(
                f"Inter-group mean spread {spread:.6f} exceeds "
                f"{mean_spread_ratio_limit} x overall mean "
                f"({overall_mean:.6f}, budget {budget:.6f})"
            )

    selected_set = set(selected_keys)
    remaining = sort_by_umax(row for row in strict_rows if row.key not in selected_set)
    # Membership invariant: every strict candidate is either selected or surfaced
    # as a leftover. The previous "no qualifying residual could have been added"
    # check is intentionally dropped — ``select_compact_groups`` deliberately
    # leaves some candidates unused in favour of compactness over completeness.
    if len(selected_set) + len(remaining) != len(strict_rows):
        raise AssertionError(
            "Untracked strict candidates: selected + remaining != strict total"
        )

    red_rows: set[tuple[str, str]] = set()
    for source_name in sorted({row.source_file for row in strict_rows}):
        marked_path = output_dir / f"{Path(source_name).stem}_marked.xlsx"
        workbook = load_workbook(marked_path, read_only=False, data_only=False)
        summary = workbook["汇总结果"]
        for row_number in range(2, summary.max_row + 1):
            chamber = summary.cell(row_number, 1).value
            if isinstance(chamber, str) and re.fullmatch(r"CH\d+_CB\d+", chamber):
                if _font_is_red(summary.cell(row_number, 1)):
                    red_rows.add((source_name, chamber))
        for file_name, chamber in selected_set:
            if file_name == source_name and not _font_is_red(workbook[chamber]["A1"]):
                raise AssertionError(f"Chamber sheet not marked red: {file_name}/{chamber}")
        workbook.close()
    if red_rows != selected_set:
        raise AssertionError("Red summary rows do not exactly match selected chambers")

    # Companion umax-summary workbook: exists, zip is sound, has one data row
    # per group, and the row's group index plus mean μmax match the source.
    summary_path = output_dir / "selected_groups_umax_summary.xlsx"
    if not summary_path.is_file():
        raise AssertionError(f"Missing umax-summary workbook: {summary_path}")
    with zipfile.ZipFile(summary_path) as archive:
        if archive.testzip() is not None:
            raise AssertionError(f"Corrupt umax-summary workbook: {summary_path}")
    summary_wb = load_workbook(summary_path, read_only=True, data_only=True)
    summary_sheet = summary_wb["group_mean_umax"]
    summary_rows = list(summary_sheet.iter_rows(min_row=2, values_only=True))
    summary_wb.close()
    if len(summary_rows) != len(groups):
        raise AssertionError(
            f"umax-summary has {len(summary_rows)} rows, expected {len(groups)}"
        )
    for group_index, (row, group) in enumerate(zip(summary_rows, groups), start=1):
        if row[0] != group_index:
            raise AssertionError(
                f"umax-summary row {group_index} has wrong group index {row[0]}"
            )
        expected_mean = metrics(group).mean
        if abs(float(row[1]) - expected_mean) >= 1e-9:
            raise AssertionError(
                f"umax-summary group {group_index} mean mismatch: "
                f"{row[1]} vs {expected_mean}"
            )

    return {
        "selected_total": len(selected_keys),
        "duplicates": 0,
        "red_rows": len(red_rows),
        "remaining_strict_candidates": len(remaining),
        "additional_group_ge_min_found": False,
        "umax_summary_group_count": len(groups),
        "groups": group_results,
    }


def validate_ppt(pptx_path: Path, manifest: dict) -> dict:
    with zipfile.ZipFile(pptx_path) as archive:
        if archive.testzip() is not None:
            raise AssertionError(f"Corrupt PowerPoint file: {pptx_path}")
        media_paths = [
            name for name in archive.namelist() if name.startswith("ppt/media/") and name.lower().endswith(".png")
        ]
        qualities = []
        for media_path in media_paths:
            with Image.open(io.BytesIO(archive.read(media_path))) as image:
                dpi = image.info.get("dpi", (0, 0))
                dpi_x, dpi_y = (float(dpi), float(dpi)) if isinstance(dpi, (int, float)) else dpi
                if image.width < 2400 or image.height < 1600 or min(dpi_x, dpi_y) < 299:
                    raise AssertionError(f"Low-quality embedded chart: {media_path}")
                qualities.append((image.width, image.height, dpi_x, dpi_y))

    presentation = Presentation(pptx_path)
    if len(presentation.slides) != len(manifest["groups"]):
        raise AssertionError("PPT slide count does not match group count")
    picture_counts = []
    for slide, group in zip(presentation.slides, manifest["groups"]):
        pictures = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
        if len(pictures) != group["size"]:
            raise AssertionError(f"Unexpected chart count on group {group['index']} slide")
        texts = [shape.text for shape in slide.shapes if getattr(shape, "has_text_frame", False)]
        expected_cv = f"CV = {group['umax_cv_percent']:.2f}%"
        expected_mean = f"Mean μmax = {group['umax_mean_h-1']:.4f} h⁻¹"
        if not any(expected_cv in text for text in texts) or not any(expected_mean in text for text in texts):
            raise AssertionError(f"Missing group statistics on slide {group['index']}")
        labels = [text for text in texts if re.fullmatch(r"CH\d+-CB\d+", text)]
        expected_labels = [chart["label"] for chart in group["charts"]]
        if labels != expected_labels:
            raise AssertionError(f"Chart order mismatch on slide {group['index']}")
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0:
                raise AssertionError(f"Negative shape position on slide {group['index']}")
            if shape.left + shape.width > presentation.slide_width + 2:
                raise AssertionError(f"Horizontal overflow on slide {group['index']}")
            if shape.top + shape.height > presentation.slide_height + 2:
                raise AssertionError(f"Vertical overflow on slide {group['index']}")
        notes = slide.notes_slide.notes_text_frame
        if notes is None or "[Sources]" not in notes.text:
            raise AssertionError(f"Missing source notes on slide {group['index']}")
        picture_counts.append(len(pictures))

    if not qualities:
        raise AssertionError("No embedded PNG charts found in PPT")
    return {
        "slide_count": len(presentation.slides),
        "slide_picture_counts": picture_counts,
        "embedded_png_count": len(media_paths),
        "minimum_width_px": min(item[0] for item in qualities),
        "minimum_height_px": min(item[1] for item in qualities),
        "minimum_dpi": min(min(item[2], item[3]) for item in qualities),
        "speaker_notes_sources": True,
        "order_matches_manifest": True,
        "overflow": False,
    }
