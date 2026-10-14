"""Smoke-test for select_compact_groups / _greedy_split_core.

Bypasses the package __init__ so openpyxl/PIL/pptx are not required.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

# Locate the package relative to this file so the test survives a project move.
PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_ROOT / "data_analysis_batch"

# IMPORTANT: do NOT ``import data_analysis_batch`` here — the package __init__
# pulls in openpyxl/PIL/pptx which are not installed in this interpreter. We
# build a stub package and load only the pure-Python submodules we need.


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Build a stub package that doesn't import the heavy modules via __init__.py.
pkg = types.ModuleType("data_analysis_batch")
pkg.__path__ = [str(PACKAGE_DIR)]
sys.modules["data_analysis_batch"] = pkg

models = _load_module("data_analysis_batch.models", PACKAGE_DIR / "models.py")
grouping = _load_module("data_analysis_batch.grouping", PACKAGE_DIR / "grouping.py")

# Re-export under plain names for ergonomics.
ChamberResult = models.ChamberResult
GroupStats = models.GroupStats

select_compact_groups = grouping.select_compact_groups
_greedy_split_core = grouping._greedy_split_core
allocate_groups = grouping.allocate_groups
_legacy_allocate_groups = grouping._legacy_allocate_groups


def make_row(source_file: str, flow_no: int, chamber_no: int, umax: float) -> ChamberResult:
    return ChamberResult(
        source_file=source_file,
        summary_row=0,
        chamber=f"CH{flow_no}_CB{chamber_no}",
        flow=f"CH{flow_no}",
        flow_no=flow_no,
        chamber_no=chamber_no,
        umax=umax,
        k=1.0,
        n0=1.0,
        r2=0.99,
        td=0.0,
        f=0.0,
        eta=0.0,
        lag=0.0,
        log=0.0,
        stable=0.0,
    )


def compute_stats(rows):
    values = [r.umax for r in rows]
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    m2 = sum((v - mean) ** 2 for v in values) / n
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    skew = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 ** 2)
    jb = n / 6 * (skew ** 2 + (kurtosis - 3) ** 2 / 4)
    p = math.exp(-jb / 2)
    return GroupStats(mean=mean, sample_sd=sd, cv=sd / mean, jb=jb, p_value=p)


def report(label: str, groups, remaining) -> None:
    sizes = [len(g) for g in groups]
    print(f"  {label}: groups={len(groups)} sizes={sizes} remaining={len(remaining)}")
    for i, g in enumerate(groups, start=1):
        stats = compute_stats(g)
        print(
            f"    group{i:>2}: n={len(g):>2} mean={stats.mean:.4f} "
            f"cv={stats.cv:.4f} p={stats.p_value:.3f} "
            f"umax∈[{min(r.umax for r in g):.4f}, {max(r.umax for r in g):.4f}]"
        )


def check_shape(groups, remaining, total, label):
    """Shape contract: leading groups == 18, trailing in [12, 18], no loss."""
    sizes = [len(g) for g in groups]
    assert all(s == 18 for s in sizes[:-1]), f"{label}: leading groups must be 18, got {sizes}"
    assert 12 <= sizes[-1] <= 18, f"{label}: trailing group must be in [12, 18], got {sizes}"
    assert sum(sizes) + len(remaining) == total, f"{label}: membership invariant broken"
    for g in groups:
        st = compute_stats(g)
        assert st.cv < 0.15 and st.p_value > 0.05, f"{label}: group failed cv={st.cv} p={st.p_value}"
    return sizes


def main() -> None:
    print("=== Test 1: 60-row compact band (uniform μmax in [0.500, 0.520]) ===")
    rows = [
        make_row(f"src_{i // 12}.xlsx", flow_no=(i // 12) + 1, chamber_no=(i % 12) + 1,
                 umax=0.500 + i * 0.00034)
        for i in range(60)
    ]
    groups, remaining = select_compact_groups(rows, max_size=18, min_size=12)
    report("select_compact_groups", groups, remaining)
    sizes = check_shape(groups, remaining, 60, "T1")
    assert len(groups) == 3, f"expected 3 groups, got {len(groups)}"
    # NOTE: the DP may legitimately prefer [18, 18, 12] over [18, 18, 18] when
    # skipping low edge rows tightens the inter-group mean spread.

    print()
    print("=== Test 2: 50-row cluster via allocate_groups ===")
    small = rows[:50]
    groups, remaining, full_count = allocate_groups(small, min_group_size=12, max_group_size=18)
    report("allocate_groups", groups, remaining)
    print(f"  full_group_count returned: {full_count}")
    assert all(12 <= len(g) <= 18 for g in groups), "all groups must be in [12, 18]"
    assert len(remaining) == 50 - sum(len(g) for g in groups), "remaining count must match"

    print()
    print("=== Test 3: 56-row compact band → 3 full groups + 2 leftover (2 < min_size) ===")
    rows56 = [
        make_row(f"s_{i}.xlsx", 1, i + 1, 0.500 + i * 0.00033)
        for i in range(56)
    ]
    groups, remaining = select_compact_groups(rows56, max_size=18, min_size=12)
    report("select_compact_groups", groups, remaining)
    sizes = check_shape(groups, remaining, 56, "T3")
    # 56 rows: max possible is 3 groups (18+18+18=54, or 18+18+12-17 tail).
    assert len(groups) == 3, f"expected three groups, got sizes={sizes}"

    print()
    print("=== Test 4: 60-row where tail fits 12 (18+18+12+12 wouldn't add up; just try 60→3groups) ===")
    # Already done in Test 1; now try 49-row → 18+18+13
    rows49 = [
        make_row(f"q_{i}.xlsx", 1, i + 1, 0.500 + i * 0.00033)
        for i in range(49)
    ]
    groups, remaining = select_compact_groups(rows49, max_size=18, min_size=12)
    sizes = [len(g) for g in groups]
    print(f"  49-row test: sizes={sizes} remaining={len(remaining)}")
    check_shape(groups, remaining, 49, "T4")
    assert len(groups) == 3, f"expected 3 groups (18+18+12..13), got {sizes}"

    print()
    print("=== Test 5: bimodal data — algorithm should still produce groups ===")
    left  = [make_row(f"a_{i}.xlsx", 1, i + 1, 0.500 + i * 0.0005) for i in range(18)]
    right = [make_row(f"b_{i}.xlsx", 2, i + 1, 0.550 + i * 0.0005) for i in range(18)]
    bimodal = sorted(left + right, key=lambda r: r.umax)
    groups, remaining = select_compact_groups(bimodal, max_size=18, min_size=12)
    report("select_compact_groups (bimodal)", groups, remaining)
    assert 1 <= len(groups) <= 2, "bimodal should yield 1-2 groups"

    print()
    print("=== Test 6: legacy should still work for backward compat ===")
    legacy_groups, legacy_remaining, legacy_window = _legacy_allocate_groups(rows[:50])
    print(f"  legacy groups: {len(legacy_groups)} sizes={[len(g) for g in legacy_groups]} "
          f"remaining={len(legacy_remaining)} window_count={legacy_window}")
    # On uniform dense data legacy produces 18+18+14 (3 groups, 0 leftover).
    # Just check it still runs and returns the expected structure.
    assert sum(len(g) for g in legacy_groups) + len(legacy_remaining) == 50
    assert all(8 <= len(g) <= 18 for g in legacy_groups), "legacy sizes in [8, 18]"

    print()
    print("=== Test 7: user's real L210-N300-IC10% data (53 strict candidates) ===")
    # umax values taken from the user's partition_results.json
    real_umax = [
        0.0691, 0.0784, 0.0803, 0.0837, 0.0859, 0.0921, 0.0926, 0.0928,
        0.0938, 0.0943, 0.0948, 0.0951, 0.0958, 0.0967, 0.0967, 0.0991,
        0.0995, 0.1003, 0.1013, 0.1066, 0.1096, 0.1121, 0.1149, 0.1159,
        0.1200, 0.1212, 0.1258, 0.1272, 0.1293, 0.1371, 0.1372, 0.1391,
        0.1519, 0.1539, 0.1556, 0.1576, 0.1606, 0.1612, 0.1645, 0.1663,
        0.1665, 0.1680, 0.1705, 0.1815, 0.1816, 0.1836, 0.1837, 0.1908,
        0.1948, 0.2042, 0.2154, 0.2461, 0.3498,
    ]
    real_rows = [
        make_row(f"real_{i}.xlsx", 1, i + 1, v) for i, v in enumerate(real_umax)
    ]
    groups, remaining = select_compact_groups(real_rows, max_size=18, min_size=12)
    report("select_compact_groups (REAL)", groups, remaining)
    sizes = [len(g) for g in groups]
    means = [compute_stats(g).mean for g in groups]
    print(f"  group means: {[f'{m:.4f}' for m in means]}")
    print(f"  mean spread (max-min): {max(means) - min(means):.4f}")
    # Shape contract: all leading groups == 18, trailing in [12, 18]
    assert all(s == 18 for s in sizes[:-1]), f"leading groups must be 18, got {sizes}"
    assert 12 <= sizes[-1] <= 18, f"trailing group must be in [12, 18], got {sizes}"
    assert len(groups) >= 2, f"expected at least 2 groups on real data, got {sizes}"
    # Every group must independently qualify
    for g in groups:
        st = compute_stats(g)
        assert st.cv < 0.15 and st.p_value > 0.05, f"group failed: cv={st.cv}, p={st.p_value}"
    # Membership invariant
    assert sum(sizes) + len(remaining) == 53

    print()
    print("=== Test 8: real data + max_mean_spread_ratio=0.3 (spread budget beats count) ===")
    groups, remaining = select_compact_groups(
        real_rows, max_size=18, min_size=12, max_mean_spread_ratio=0.3
    )
    report("ratio=0.3", groups, remaining)
    means = [compute_stats(g).mean for g in groups]
    overall = sum(r.umax for r in real_rows) / len(real_rows)
    spread = max(means) - min(means)
    print(f"  means={[f'{m:.4f}' for m in means]}  spread={spread:.4f}  budget={0.3 * overall:.4f}")
    check_shape(groups, remaining, 53, "T8")
    assert spread <= 0.3 * overall + 1e-12, f"spread {spread} exceeds budget {0.3 * overall}"

    print()
    print("=== Test 9: real data + huge ratio (10.0) == unconstrained behaviour ===")
    groups2, remaining2 = select_compact_groups(
        real_rows, max_size=18, min_size=12, max_mean_spread_ratio=10.0
    )
    sizes2 = [len(g) for g in groups2]
    print(f"  sizes={sizes2}")
    check_shape(groups2, remaining2, 53, "T9")
    assert len(groups2) == 3, f"huge ratio should match unconstrained count, got {sizes2}"

    print()
    print("=== Test 10: uniform data + ratio 0.3 keeps 3 groups (already within budget) ===")
    groups3, remaining3 = select_compact_groups(
        rows, max_size=18, min_size=12, max_mean_spread_ratio=0.3
    )
    sizes3 = [len(g) for g in groups3]
    print(f"  sizes={sizes3} remaining={len(remaining3)}")
    assert len(groups3) == 3, f"tight uniform data should still yield 3 groups, got {sizes3}"

    print()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
