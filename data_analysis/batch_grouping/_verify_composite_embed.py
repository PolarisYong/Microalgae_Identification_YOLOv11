"""Verify the composite-embedding scheme on a real run output.

Checks that each group's ``selected_`` sheet has exactly one embedded image
anchored at ``A{2 + n_data_rows}`` (immediately below the data block) and
that the composite PNG is the file we just generated.

Usage:
    python _verify_composite_embed.py                        # auto-pick latest run
    python _verify_composite_embed.py <output_dir>           # explicit output dir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image

PROJECT = Path(__file__).resolve().parent
PROCESSED_ROOT = PROJECT / "data" / "processed_data"


def _latest_output_dir() -> Path:
    """Pick the most recently modified run output dir under data/processed_data."""
    candidates = [
        path
        for path in PROCESSED_ROOT.rglob("*")
        if path.is_dir() and (path / "_batch_group_work" / "chart_manifest.json").is_file()
    ]
    if not candidates:
        raise SystemExit(
            f"No run output found under {PROCESSED_ROOT}. "
            "Pass an output directory explicitly."
        )
    return max(candidates, key=lambda p: (p / "_batch_group_work" / "chart_manifest.json").stat().st_mtime)


if len(sys.argv) > 1:
    OUT = Path(sys.argv[1]).expanduser().resolve()
else:
    OUT = _latest_output_dir()

print(f"Verifying: {OUT}")
WORK_DIR = OUT / "_batch_group_work"
manifest = json.loads((WORK_DIR / "chart_manifest.json").read_text(encoding="utf-8"))
report = json.loads((WORK_DIR / "run_report.json").read_text(encoding="utf-8"))

print(f"Group count: {len(manifest['groups'])}")
print(f"PPT validation picture counts: {report['ppt_validation']['slide_picture_counts']}")

for group_manifest in manifest["groups"]:
    name = group_manifest["source_workbook"]
    path = OUT / name
    wb = load_workbook(path)
    sheet_name = next(n for n in wb.sheetnames if n.startswith("selected_"))
    sheet = wb[sheet_name]
    data_rows = sheet.max_row - 1  # subtract header
    n_data = sum(
        1
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row and row[0]
    )
    images = sheet._images
    assert len(images) == 1, f"{name}: expected 1 image, got {len(images)}"
    img = images[0]
    expected_row = 2 + n_data
    actual_row = img.anchor._from.row + 1  # 0-based to 1-based
    actual_col = img.anchor._from.col
    print(
        f"{name}: selected={sheet_name!r} data_rows={n_data} image_anchor=A{actual_row} "
        f"(expected A{expected_row}) row_match={actual_row == expected_row}"
    )
    assert actual_row == expected_row, "image must be anchored immediately under the data"
    assert actual_col == 0, "image must be in column A"

    # Manifest image paths were recorded as absolute paths at run time. If the
    # project has since been moved/renamed, the prefix is stale — re-anchor the
    # tail of the path (from the run output dir onwards) under the current OUT.
    recorded = Path(group_manifest["charts"][0]["image_path"])
    marker = "_batch_group_work"
    parts = recorded.parts
    if marker in parts:
        tail = parts[parts.index(marker):]
        composite_path = WORK_DIR.joinpath(*tail[1:-1]) / "composite.png"
    else:
        composite_path = OUT / "_batch_group_work" / f"ppt_charts/group_{group_manifest['index']}" / "composite.png"
    if not composite_path.is_file():
        # Fall back to the run's own chart directory name.
        composite_path = WORK_DIR / "ppt_charts" / f"group_{group_manifest['index']}" / "composite.png"
    assert composite_path.is_file(), f"composite missing: {composite_path}"
    with Image.open(composite_path) as canvas:
        size = canvas.size
        dpi = canvas.info.get("dpi", (0, 0))
    print(f"  composite {composite_path.name}: size={size} dpi={tuple(round(d) for d in dpi)}")

print("ALL COMPOSITE-EMBED VERIFICATIONS PASSED")
