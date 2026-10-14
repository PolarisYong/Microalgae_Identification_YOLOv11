"""Standalone smoke-test for ppt_io.compose_group_growth_chart."""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

# Locate the package relative to this file so the test survives a project move.
PROJECT = Path(__file__).resolve().parent
PACKAGE = PROJECT / "data_analysis_batch"

# Avoid pulling the package __init__ (heavy imports); stub the package.
pkg = types.ModuleType("data_analysis_batch")
pkg.__path__ = [str(PACKAGE)]
sys.modules["data_analysis_batch"] = pkg


def _load(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


models = _load("data_analysis_batch.models", PACKAGE / "models.py")
grouping = _load("data_analysis_batch.grouping", PACKAGE / "grouping.py")
ppt_io = _load("data_analysis_batch.ppt_io", PACKAGE / "ppt_io.py")

from PIL import Image, ImageDraw

tmp = Path(tempfile.mkdtemp(prefix="ppt_io_compose_test_"))

# Make 12 fake chamber PNGs.
charts_12 = []
for i in range(12):
    p = tmp / f"{i + 1:02d}_CH1-CB{i + 1}.png"
    img = Image.new("RGB", (2400, 1600), (250, 250, 252))
    d = ImageDraw.Draw(img)
    pts = [(int(x * 40), int(1500 - (math.sin(x / 3.0 + i) * 200 + 1100 - x * 5))) for x in range(50)]
    d.line(pts, fill=(31, 41, 55), width=8)
    img.save(p, dpi=(300, 300))
    charts_12.append({"label": f"CH1-CB{i + 1}", "image_path": str(p)})

# 12-chamber composite with title.
composite_12 = ppt_io.compose_group_growth_chart(
    charts_12,
    tmp / "composite_12.png",
    title="Group 1 (n = 12) — Mean mu_max = 0.5555",
    grid_rows=2,
)
with Image.open(composite_12) as img:
    cols = math.ceil(12 / 2)
    expected_w = cols * (720 + 2 * 16) + (cols - 1) * 10
    title_h = 36 + 10
    expected_h = title_h + 2 * (32 + 480 + 2 * 16) + (2 - 1) * 10
    assert img.size == (expected_w, expected_h), f"size mismatch: {img.size}, expected {(expected_w, expected_h)}"
    assert min(img.info["dpi"]) >= 299, f"dpi mismatch: {img.info.get('dpi')}"
print(f"[OK] 12-chamber composite {img.size} dpi={img.info.get('dpi')}")

# 18-chamber composite (no title).
charts_18 = []
for i in range(18):
    p = tmp / f"g2_{i + 1:02d}_CH2-CB{i + 1}.png"
    img = Image.new("RGB", (2400, 1600), (252, 250, 250))
    d = ImageDraw.Draw(img)
    pts = [(int(x * 40), int(1500 - (math.cos(x / 2.5 + i) * 200 + 1100 - x * 5))) for x in range(50)]
    d.line(pts, fill=(17, 24, 39), width=8)
    img.save(p, dpi=(300, 300))
    charts_18.append({"label": f"CH2-CB{i + 1}", "image_path": str(p)})

composite_18 = ppt_io.compose_group_growth_chart(charts_18, tmp / "composite_18.png", grid_rows=2)
with Image.open(composite_18) as img:
    cols18 = math.ceil(18 / 2)
    e_w = cols18 * (720 + 32) + (cols18 - 1) * 10
    e_h = 2 * (32 + 480 + 32) + 10
    assert img.size == (e_w, e_h), f"size mismatch: {img.size}, expected {(e_w, e_h)}"
print(f"[OK] 18-chamber composite {img.size}")

# Missing PNG is silently skipped.
ghosts = charts_12 + [{"label": "GHOST", "image_path": str(tmp / "does_not_exist.png")}]
ghost_out = ppt_io.compose_group_growth_chart(ghosts, tmp / "composite_ghost.png", grid_rows=2)
with Image.open(ghost_out) as img:
    # No title this time: compute the no-title height.
    no_title_h = 2 * (32 + 480 + 2 * 16) + (2 - 1) * 10
    cols12 = math.ceil(12 / 2)
    no_title_w = cols12 * (720 + 2 * 16) + (cols12 - 1) * 10
    assert img.size == (no_title_w, no_title_h), f"ghost size mismatch: {img.size}"
print(f"[OK] ghost PNG skipped; composite still {img.size}")

# Empty input -> RuntimeError.
try:
    ppt_io.compose_group_growth_chart([], tmp / "empty.png", grid_rows=2)
except RuntimeError as e:
    print(f"[OK] empty input raises: {e}")

print("ALL PPT-IO COMPOSE TESTS PASSED")
