from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_analysis_batch import PipelineConfig, run_pipeline


# 在这里直接填写需要批处理的目标文件夹和对应输出文件夹。
# 两个列表按位置一一对应且长度必须相同。输出路径填空字符串时，
# 结果写回对应目标文件夹。
input_data = "L30‑N20‑IC0.5%"
output_data = input_data
output_data_label = "18"

# 分组数量规则：优先尽可能多地生成 TARGET_MAX_GROUP_SIZE 条的完整组，
# 然后从剩余数据中迭代筛选不少于 TARGET_MIN_GROUP_SIZE 条的合格组。
TARGET_MAX_GROUP_SIZE: int = 18
TARGET_MIN_GROUP_SIZE: int = 12

# 组间均值极差约束（可选，默认关闭）：以“严格候选 μmax 总体均值”的倍数给出，
# 例如 0.3 表示任意两组 μmax 均值之差不得超过 0.3 × 总体均值。
# 开启后“组间差距小”优先于“组数最多”；设为 None 恢复组数优先的原行为。
# 命令行 --max-mean-spread-ratio 优先于此处的取值。
TARGET_MAX_MEAN_SPREAD_RATIO: float | None = None


TARGET_BASE_DIRS: list[str] = [
    rf"F:\Workbuddy_Workspace\Peipei_Workspace\实验结果汇总\{input_data}",
]
TARGET_OUTPUT_DIRS: list[str] = [
    rf"F:\Workbuddy_Workspace\Peipei_Workspace\实验结果汇总\{output_data}\{output_data_label}",
]


def _default_reference(base_dir: Path) -> Path:
    candidates = [base_dir.parent / "参考输出.xlsx", base_dir / "参考输出.xlsx"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-Python data analysis, Excel marking, and 2x9 PPT generation."
    )
    parser.add_argument(
        "--base-dir",
        dest="base_dirs",
        action="append",
        type=Path,
        help=(
            "Folder containing original source XLSX files. Repeat this option to process multiple folders. "
            "When omitted, TARGET_BASE_DIRS in this file is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dirs",
        action="append",
        type=Path,
        help=(
            "Output folder paired by order with --base-dir. Repeat for multiple targets. "
            "When omitted, each target folder is used."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference workbook; defaults to 参考输出.xlsx in each target's parent folder",
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        help="Preferred full-group size; defaults to TARGET_MAX_GROUP_SIZE in this file",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        help="Minimum accepted residual-group size; defaults to TARGET_MIN_GROUP_SIZE in this file",
    )
    parser.add_argument(
        "--max-mean-spread-ratio",
        type=float,
        default=None,
        help=(
            "Optional cap on the inter-group mean spread as a fraction of the overall "
            "strict-candidate mean, e.g. 0.3 means any two group means may differ by at "
            "most 0.3 x the overall mean. When set, packings whose group means stay within "
            "this budget are preferred even if they yield fewer groups. "
            "Defaults to TARGET_MAX_MEAN_SPREAD_RATIO in this file."
        ),
    )
    parser.add_argument(
        "--ndjson-policy",
        choices=("archive", "delete", "keep"),
        default="archive",
        help="How to handle root-level .ndjson files (default: archive under _batch_group_work/logs)",
    )
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> int:
    args = parse_args()
    max_group_size = args.max_group_size if args.max_group_size is not None else TARGET_MAX_GROUP_SIZE
    min_group_size = args.min_group_size if args.min_group_size is not None else TARGET_MIN_GROUP_SIZE
    max_mean_spread_ratio = (
        args.max_mean_spread_ratio
        if args.max_mean_spread_ratio is not None
        else TARGET_MAX_MEAN_SPREAD_RATIO
    )
    if min_group_size < 2 or max_group_size <= min_group_size or max_group_size > 18:
        print(
            "ERROR: Group sizes must satisfy 2 <= min group size < max group size <= 18.",
            file=sys.stderr,
        )
        return 2

    using_cli_targets = bool(args.base_dirs)
    targets = args.base_dirs or [Path(value) for value in TARGET_BASE_DIRS if value.strip()]
    if not targets:
        print(
            "ERROR: No target folder was specified. Edit TARGET_BASE_DIRS or use --base-dir.",
            file=sys.stderr,
        )
        return 2

    if using_cli_targets:
        if args.output_dirs and len(args.output_dirs) != len(targets):
            print(
                "ERROR: The number of --output-dir values must match the number of --base-dir values.",
                file=sys.stderr,
            )
            return 2
        output_targets: list[Path | None] = args.output_dirs or [None] * len(targets)
    else:
        if args.output_dirs:
            print("ERROR: --output-dir must be used together with --base-dir.", file=sys.stderr)
            return 2
        if len(TARGET_OUTPUT_DIRS) != len(targets):
            print(
                "ERROR: TARGET_OUTPUT_DIRS and TARGET_BASE_DIRS must contain the same number of paths.",
                file=sys.stderr,
            )
            return 2
        output_targets = [Path(value) if value.strip() else None for value in TARGET_OUTPUT_DIRS]

    if len(output_targets) != len(targets):
        print("ERROR: Each target folder must have exactly one output-path setting.", file=sys.stderr)
        return 2

    reports: list[dict] = []
    failures: list[dict[str, str]] = []
    for index, (target, output_target) in enumerate(zip(targets, output_targets), start=1):
        base_dir = _resolved(target)
        output_dir = _resolved(output_target) if output_target else base_dir
        reference = _resolved(args.reference) if args.reference else _default_reference(base_dir).resolve()
        print(f"[{index}/{len(targets)}] Input: {base_dir}", flush=True)
        print(f"[{index}/{len(targets)}] Output: {output_dir}", flush=True)
        print(
            f"[{index}/{len(targets)}] Group sizes: preferred={max_group_size}, minimum={min_group_size}",
            flush=True,
        )
        if max_mean_spread_ratio is not None:
            print(
                f"[{index}/{len(targets)}] Max mean spread ratio: {max_mean_spread_ratio}",
                flush=True,
            )
        config = PipelineConfig(
            base_dir=base_dir,
            output_dir=output_dir,
            reference_path=reference,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            ndjson_policy=args.ndjson_policy,
            max_mean_spread_ratio=max_mean_spread_ratio,
        )
        try:
            reports.append(run_pipeline(config))
        except Exception as exc:
            failures.append({"base_dir": str(base_dir), "error": str(exc)})
            print(f"ERROR [{base_dir}]: {exc}", file=sys.stderr, flush=True)

    if len(reports) == 1 and not failures:
        payload: dict = reports[0]
    else:
        payload = {
            "implementation": "pure-python",
            "requested_count": len(targets),
            "successful_count": len(reports),
            "failed_count": len(failures),
            "results": reports,
            "failures": failures,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
