from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .excel_io import (
    build_group_workbooks,
    clean_generated_outputs,
    collect_candidates,
    discover_source_workbooks,
    embed_chamber_images,
    handle_root_ndjson,
    mark_source_workbooks,
    write_partition_results,
    write_umax_summary,
)
from .grouping import allocate_groups, metrics
from .ppt_io import build_growth_curve_ppt, extract_growth_charts
from .validation import validate_excel_outputs, validate_ppt


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    output_dir: Path
    reference_path: Path
    min_group_size: int = 8
    max_group_size: int = 18
    cv_limit: float = 0.15
    normality_p_limit: float = 0.05
    ndjson_policy: str = "archive"
    max_mean_spread_ratio: float | None = None

    def validate(self) -> None:
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"Base directory does not exist: {self.base_dir}")
        if not self.reference_path.is_file():
            raise FileNotFoundError(f"Reference workbook does not exist: {self.reference_path}")
        if (
            self.min_group_size < 2
            or self.max_group_size <= self.min_group_size
            or self.max_group_size > 18
        ):
            raise ValueError("Group sizes must satisfy 2 <= min_group_size < max_group_size <= 18")
        if self.ndjson_policy not in {"archive", "delete", "keep"}:
            raise ValueError("ndjson_policy must be archive, delete, or keep")
        if self.max_mean_spread_ratio is not None and self.max_mean_spread_ratio <= 0:
            raise ValueError("max_mean_spread_ratio must be positive when provided")


def run_pipeline(config: PipelineConfig) -> dict:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = config.output_dir / "_batch_group_work"
    logs_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    clean_generated_outputs(config.output_dir)
    source_paths = discover_source_workbooks(config.base_dir)
    strict_rows = collect_candidates(source_paths)
    if len(strict_rows) < config.min_group_size:
        raise RuntimeError(
            f"Only {len(strict_rows)} strict candidates were found; at least {config.min_group_size} are required"
        )

    groups, remaining, window_count = allocate_groups(
        strict_rows,
        min_group_size=config.min_group_size,
        max_group_size=config.max_group_size,
        cv_limit=config.cv_limit,
        normality_p_limit=config.normality_p_limit,
        max_mean_spread_ratio=config.max_mean_spread_ratio,
    )
    if not groups:
        raise RuntimeError("No qualifying group could be formed")

    partition_path = work_dir / "partition_results.json"
    write_partition_results(
        partition_path,
        source_paths,
        strict_rows,
        groups,
        remaining,
        window_count,
        config.min_group_size,
        config.max_group_size,
        config.cv_limit,
        config.normality_p_limit,
        config.max_mean_spread_ratio,
    )
    marked_outputs = mark_source_workbooks(
        config.base_dir,
        config.output_dir,
        source_paths,
        groups,
    )
    group_outputs = build_group_workbooks(
        config.reference_path,
        config.output_dir,
        groups,
        strict_rows,
        source_paths,
        config.min_group_size,
        config.max_group_size,
    )
    umax_summary_path = write_umax_summary(
        config.output_dir / "selected_groups_umax_summary.xlsx",
        groups,
    )
    manifest_path, manifest = extract_growth_charts(config.base_dir, work_dir, groups)
    pptx_path = build_growth_curve_ppt(
        config.output_dir / "selected_groups_growth_curves_2x9.pptx",
        manifest,
    )
    # Embed each chamber's growth-curve PNG into the matching group workbook,
    # one image per chamber, anchored in a row band directly below the data.
    for group_path, group_chart_manifest in zip(
        group_outputs, manifest["groups"]
    ):
        embed_chamber_images(group_path, group_chart_manifest["charts"])
    handle_root_ndjson(config.output_dir, logs_dir, config.ndjson_policy)

    excel_validation = validate_excel_outputs(
        config.base_dir,
        config.output_dir,
        groups,
        strict_rows,
        config.min_group_size,
        config.max_group_size,
        config.cv_limit,
        config.normality_p_limit,
        config.max_mean_spread_ratio,
    )
    ppt_validation = validate_ppt(pptx_path, manifest)
    report = {
        "implementation": "pure-python",
        "base_dir": str(config.base_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "reference_workbook": str(config.reference_path.resolve()),
        "source_files": [path.name for path in source_paths],
        "strict_count": len(strict_rows),
        "window_count": window_count,
        "min_group_size": config.min_group_size,
        "max_group_size": config.max_group_size,
        "max_mean_spread_ratio": config.max_mean_spread_ratio,
        "group_sizes": [len(group) for group in groups],
        "group_stats": [metrics(group).to_dict() for group in groups],
        "remaining_strict_count": len(remaining),
        "group_workbooks": [str(path.resolve()) for path in group_outputs],
        "marked_workbooks": [str(path.resolve()) for path in marked_outputs],
        "umax_summary": str(umax_summary_path.resolve()),
        "pptx": str(pptx_path.resolve()),
        "partition_results": str(partition_path.resolve()),
        "chart_manifest": str(manifest_path.resolve()),
        "excel_validation": excel_validation,
        "ppt_validation": ppt_validation,
    }
    report_path = work_dir / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["run_report"] = str(report_path.resolve())
    return report
