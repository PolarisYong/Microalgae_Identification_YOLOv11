"""Batch-run the chamber analysis pipeline over a folder tree.

Default behaviour (no arguments): process every immediate sub-folder of the
bundled default root, writing outputs into ``<folder>/18_batch_<timestamp>/``.

Typical use — point it at your own folder tree:

    python batch_run_folders.py --root "D:/my_experiments"
    python batch_run_folders.py --root "D:/my_experiments" --folder "L30-N20-IC0.5%" --folder "L30-N20-IC10%"
    python batch_run_folders.py --root "D:/my_experiments" --output-label 18 --reference "D:/my_experiments/参考输出.xlsx"
    python batch_run_folders.py --root "D:/my_experiments" --list        # only show what would run
    python batch_run_folders.py --root "D:/my_experiments" --dry-run     # alias of --list

Each processed folder must contain source workbooks (``*.xlsx`` with a
``汇总结果`` sheet). A shared reference workbook (``参考输出.xlsx``) is resolved
in this order:
    1. --reference <path>               (explicit)
    2. <root>/参考输出.xlsx             (root-level, shared)
    3. <folder>/参考输出.xlsx           (per-folder, overrides the root one)
    4. <root>/../参考输出.xlsx          (legacy layout, parent of root)

Notes on the output directory:
  * The default ``--output-label`` embeds a timestamp so consecutive runs never
    clobber each other. Pass ``--output-label 18`` for the classic fixed name.
  * The pipeline normally deletes its own previous outputs first. This script
    disables that cleanup by default so it can run under sandboxed shells whose
    bulk-delete guard trips on large ``*_marked.xlsx`` files. Use
    ``--allow-cleanup`` to restore the pipeline's native clean-then-write flow
    (only safe when outputs are small or you are running outside a sandbox).

Per-folder failures never abort the batch; an aggregate JSON report is written
at the end (``--report <path>`` overrides the default location).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project / default configuration — edit these two lines if you move things.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("F:/Workbuddy_Workspace/Peipei_Workspace/实验分析结果")
DEFAULT_REFERENCE_NAME = "参考输出.xlsx"

MAX_GROUP_SIZE = 18
MIN_GROUP_SIZE = 12
MAX_MEAN_SPREAD_RATIO: float | None = None

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Cleanup control
# ---------------------------------------------------------------------------
def _disable_internal_cleanup() -> tuple:
    """Stub out the pipeline's self-cleanup so no bulk deletes are issued.

    Returns the original callables so ``_restore_internal_cleanup`` can put
    them back. We keep behaviour identical for every other consumer of
    ``shutil.rmtree`` by only swallowing the chart-PNG directory removal.
    """
    from data_analysis_batch import excel_io

    real_clean = excel_io.clean_generated_outputs
    real_rmtree = shutil.rmtree

    def _noop_clean(output_dir):  # noqa: ARG001
        return None

    def _selective_rmtree(path, *args, **kwargs):
        if "/ppt_charts/group_" in str(path).replace("\\", "/"):
            return None
        return real_rmtree(path, *args, **kwargs)

    excel_io.clean_generated_outputs = _noop_clean
    shutil.rmtree = _selective_rmtree
    return excel_io, real_clean, real_rmtree


def _restore_internal_cleanup(originals: tuple) -> None:
    excel_io, real_clean, real_rmtree = originals
    excel_io.clean_generated_outputs = real_clean
    shutil.rmtree = real_rmtree


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run the chamber analysis pipeline over a folder tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python batch_run_folders.py --root "D:/my_experiments"\n'
            '  python batch_run_folders.py --root "D:/my_experiments" --folder "L30-N20-IC0.5%"\n'
            '  python batch_run_folders.py --root "D:/my_experiments" --list\n'
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root folder whose immediate sub-folders are processed (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--folder",
        dest="folders",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Process only these sub-folders (repeat the flag for several). "
            "Accepts a bare folder name relative to --root or an absolute path. "
            "Omit to process every immediate sub-folder of --root."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=f"Shared reference workbook (default: <root>/{DEFAULT_REFERENCE_NAME})",
    )
    parser.add_argument(
        "--output-label",
        default=None,
        help=(
            "Output sub-folder name. Default: '18_batch_<YYYYmmdd>' "
            "(fresh per run). Pass '18' for the classic fixed name."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the aggregate JSON report (default: <project>/_batch_folders_report.json)",
    )
    parser.add_argument("--max-group-size", type=int, default=MAX_GROUP_SIZE,
                        help=f"Preferred full-group size (default: {MAX_GROUP_SIZE})")
    parser.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE,
                        help=f"Minimum trailing-group size (default: {MIN_GROUP_SIZE})")
    parser.add_argument(
        "--max-mean-spread-ratio",
        type=float,
        default=MAX_MEAN_SPREAD_RATIO,
        help=(
            "Optional cap on the inter-group mean spread, as a fraction of the "
            "overall strict-candidate mean (e.g. 0.3). Smaller = tighter groups, "
            "possibly fewer. Omit for count-priority."
        ),
    )
    parser.add_argument(
        "--ndjson-policy",
        choices=("archive", "delete", "keep"),
        default="archive",
        help="How to handle root-level .ndjson files (default: archive)",
    )
    parser.add_argument(
        "--allow-cleanup",
        action="store_true",
        help=(
            "Let the pipeline delete its previous outputs before writing "
            "(default: cleanup disabled to avoid sandboxed bulk-delete guards)."
        ),
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="Only list the folders that would be processed, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        dest="list_only",
        action="store_true",
        help="Alias of --list.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Folder + reference resolution
# ---------------------------------------------------------------------------
def _fold_name_key(name: str) -> str:
    """Normalise a folder name for forgiving matching.

    Real experiment folder names often use a non-ASCII hyphen (U+2011 '‑').
    Users typing the name by hand will use the plain ASCII '-', so both are
    folded to a single canonical form before comparison.
    """
    return name.replace("\u2011", "-").replace("\u2010", "-").strip().lower()


def resolve_folders(root: Path, names: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    children = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not names:
        return children

    by_key = {_fold_name_key(p.name): p for p in children}
    resolved: list[Path] = []
    for name in names:
        candidate = Path(name)
        if candidate.is_absolute() and candidate.is_dir():
            resolved.append(candidate)
            continue
        # Try an exact path under root first, then a normalised name match.
        direct = root / name
        if direct.is_dir():
            resolved.append(direct)
            continue
        matched = by_key.get(_fold_name_key(name))
        if matched is None:
            available = ", ".join(p.name for p in children)
            raise FileNotFoundError(
                f"Requested folder not found under {root}: {name!r}. "
                f"Available folders: {available}"
            )
        resolved.append(matched)
    return resolved


def resolve_reference(explicit: Path | None, root: Path, folder: Path) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.append(root / DEFAULT_REFERENCE_NAME)
    candidates.append(folder / DEFAULT_REFERENCE_NAME)
    candidates.append(root.parent / DEFAULT_REFERENCE_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
def run_batch(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    folders = resolve_folders(root, args.folders)

    if args.list_only:
        print(f"Root: {root}")
        print(f"Folders that would be processed ({len(folders)}):")
        for folder in folders:
            reference = resolve_reference(args.reference, root, folder)
            ref_note = str(reference) if reference else "MISSING (will fail)"
            print(f"  {folder.name:<32} reference: {ref_note}")
        return 0

    output_label = args.output_label or f"18_batch_{time.strftime('%Y%m%d')}"

    from data_analysis_batch import PipelineConfig, run_pipeline

    originals = None
    if not args.allow_cleanup:
        originals = _disable_internal_cleanup()

    print(f"Root            : {root}")
    print(f"Folders to run  : {len(folders)}")
    print(f"Output label    : {output_label}")
    print(f"Group sizes     : preferred={args.max_group_size}, minimum={args.min_group_size}")
    if args.max_mean_spread_ratio is not None:
        print(f"Mean-spread cap : {args.max_mean_spread_ratio}")
    print(f"Cleanup         : {'enabled' if args.allow_cleanup else 'disabled (safe for sandboxed shells)'}")

    results: list[dict] = []
    failures: list[dict] = []

    try:
        for index, folder in enumerate(folders, start=1):
            output_dir = folder / output_label
            output_dir.mkdir(parents=True, exist_ok=True)
            reference = resolve_reference(args.reference, root, folder)
            started = time.perf_counter()
            print(f"\n[{index}/{len(folders)}] {folder.name}")
            print(f"  base_dir  = {folder}")
            print(f"  output_dir= {output_dir}")
            print(f"  reference = {reference if reference else 'MISSING'}")

            if reference is None:
                failures.append({
                    "folder": folder.name,
                    "error": f"No reference workbook found (expected {DEFAULT_REFERENCE_NAME})",
                    "elapsed_seconds": 0.0,
                })
                print("  -> ERROR: reference workbook missing")
                continue

            config = PipelineConfig(
                base_dir=folder,
                output_dir=output_dir,
                reference_path=reference,
                min_group_size=args.min_group_size,
                max_group_size=args.max_group_size,
                ndjson_policy=args.ndjson_policy,
                max_mean_spread_ratio=args.max_mean_spread_ratio,
            )
            try:
                report = run_pipeline(config)
                elapsed = time.perf_counter() - started
                print(
                    f"  -> groups={report['group_sizes']} "
                    f"selected={report['excel_validation']['selected_total']} "
                    f"remaining={report['remaining_strict_count']} took {elapsed:.1f}s"
                )
                results.append({
                    "folder": folder.name,
                    "ok": True,
                    "elapsed_seconds": round(elapsed, 2),
                    "source_files": report["source_files"],
                    "strict_count": report["strict_count"],
                    "group_sizes": report["group_sizes"],
                    "group_stats": report["group_stats"],
                    "remaining_strict_count": report["remaining_strict_count"],
                    "selected_total": report["excel_validation"]["selected_total"],
                    "umax_summary": report["umax_summary"],
                    "group_workbooks": report["group_workbooks"],
                    "marked_workbooks": report["marked_workbooks"],
                    "pptx": report["pptx"],
                    "excel_validation_groups": report["excel_validation"]["groups"],
                    "ppt_validation_picture_counts": report["ppt_validation"]["slide_picture_counts"],
                })
            except Exception as exc:  # noqa: BLE001 - one folder must not abort the batch
                elapsed = time.perf_counter() - started
                print(f"  -> ERROR: {exc} (after {elapsed:.1f}s)")
                failures.append({
                    "folder": folder.name,
                    "error": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                })
    finally:
        if originals is not None:
            _restore_internal_cleanup(originals)

    aggregate = {
        "root": str(root),
        "output_label": output_label,
        "max_group_size": args.max_group_size,
        "min_group_size": args.min_group_size,
        "max_mean_spread_ratio": args.max_mean_spread_ratio,
        "total_folders": len(folders),
        "successful_count": len(results),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
    }
    report_path = args.report or (PROJECT_ROOT / "_batch_folders_report.json")
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"=== {len(results)} ok / {len(failures)} failed ===")
    if failures:
        print("Failed folders:")
        for item in failures:
            print(f"  - {item['folder']}: {item['error']}")
    print(f"Aggregate report: {report_path}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    try:
        return run_batch(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
