#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微藻识别主流程 —— 一键执行入口。

按顺序串起 6 个步骤，一次配置即可跑完整条链路：

    ①  按流道分拣原始微藻图片      01_select_ch.py  /  02_move_photoes.py
    ②  识别                        detection/detect_batch.py
    ③  汇总按流道/腔室的识别数据   00_merge_excel_summary.py
    ④  标准化                     01_standardize_relative_cell_number.py
    ⑤  批次可视化结果             08_growth_curve_batch.py

（原脚本编号中 01、02 同属第 ① 步，故为 5 个执行阶段 / 6 个脚本。）

常见用法
--------
    # 跑完整流程（使用 configs/pipeline.yaml 或 --config 指定的批次配置）
    python microalgae_identification/run_pipeline.py --config configs/batch_20261001.yaml

    # 只跑识别 + 之后的步骤（图片已分拣好）
    python microalgae_identification/run_pipeline.py --from detect

    # 只跑单步
    python microalgae_identification/run_pipeline.py --only merge

    # 先看解析出的路径对不对，不实际执行
    python microalgae_identification/run_pipeline.py --dry-run

    # 临时换批次，不写配置文件
    python microalgae_identification/run_pipeline.py --date 20261015 --hours 96

每一步失败即中断（fail-fast），便于定位是数据没到位还是配置写错。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from config import add_config_arguments, load_config  # noqa: E402

# 阶段名 → (显示名, 脚本相对路径)。顺序即执行顺序。
STAGES: list[tuple[str, str, str]] = [
    ("sort", "①-1 按流道分拣原始图（周目录重排）", "preprocessing/01_select_ch.py"),
    ("move", "①-2 按流道归集并重命名图片", "preprocessing/02_move_photoes.py"),
    ("detect", "② 微藻识别（YOLO 分割）", "detection/detect_batch.py"),
    ("merge", "③ 汇总按流道/腔室的识别数据", "postprocessing/00_merge_excel_summary.py"),
    ("standardize", "④ 相对细胞数量标准化", "postprocessing/01_standardize_relative_cell_number.py"),
    ("visualize", "⑤ 批次生长曲线与可视化结果", "postprocessing/08_growth_curve_batch.py"),
]
STAGE_NAMES = [name for name, _, _ in STAGES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="微藻识别主流程一键执行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arguments(parser)
    parser.add_argument("--from", dest="start_from", choices=STAGE_NAMES,
                        help="从指定阶段开始（默认从头）")
    parser.add_argument("--to", dest="stop_at", choices=STAGE_NAMES,
                        help="执行到指定阶段为止（含）")
    parser.add_argument("--only", dest="only_stage", choices=STAGE_NAMES,
                        help="只执行指定阶段，等价于 --from X --to X")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要执行的命令与配置，不实际运行")
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGE_NAMES,
                        help="跳过指定阶段（如 --skip sort move，图片已手工就位时常用）")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="某阶段失败时继续跑后续阶段（默认失败即中断）")
    parser.add_argument("--extra-args", default="",
                        help="追加给每个子脚本的命令行参数，如 \"--channels 1-6\"")
    return parser


def resolve_stage_range(args) -> list[str]:
    if args.only_stage:
        return [args.only_stage]
    start = STAGE_NAMES.index(args.start_from) if args.start_from else 0
    end = STAGE_NAMES.index(args.stop_at) if args.stop_at else len(STAGE_NAMES) - 1
    if end < start:
        raise SystemExit(f"--to（{args.stop_at}）在 --from（{args.start_from}）之前，无法执行")
    return [name for name in STAGE_NAMES[start:end + 1] if name not in set(args.skip)]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 复用同一套配置参数解析，得到各步骤共享的批次配置
    _config_parser = argparse.ArgumentParser(add_help=False)
    add_config_arguments(_config_parser)
    cfg = load_config(argv, parser=_config_parser)

    stages = resolve_stage_range(args)
    print("=" * 68)
    print("微藻识别主流程")
    print("=" * 68)
    print(cfg.describe())
    print("-" * 68)
    print("待执行阶段：")
    for name in stages:
        label = next(lbl for n, lbl, _ in STAGES if n == name)
        print(f"  · {label}")
    print("=" * 68)

    if args.dry_run:
        print("\n[dry-run] 未执行任何脚本。完整命令示例：")
        for name in stages:
            script = next(s for n, _, s in STAGES if n == name)
            print(f"  python {PACKAGE_ROOT.name}/{script} --config {cfg.config_path or 'configs/pipeline.yaml'}")
        return 0

    started = time.time()
    for index, name in enumerate(stages, start=1):
        label, script_rel = next((lbl, s) for n, lbl, s in STAGES if n == name)
        script = PACKAGE_ROOT / script_rel
        if not script.exists():
            print(f"\n[错误] 脚本不存在：{script}")
            return 2

        cmd = [sys.executable, str(script)]
        if cfg.config_path:
            cmd += ["--config", cfg.config_path]
        else:
            cmd += ["--date", cfg.batch_date, "--photo-root", cfg.photo_root,
                    "--model", cfg.model_weights, "--hours", cfg.valid_hours]
        if args.extra_args:
            cmd += args.extra_args.split()

        print(f"\n[{index}/{len(stages)}] {label}")
        print(f"    $ {' '.join(cmd)}")
        print("-" * 68)
        result = subprocess.run(cmd, cwd=str(PACKAGE_ROOT))
        if result.returncode != 0:
            print(f"\n[失败] 阶段 {name} 退出码 {result.returncode}")
            if not args.continue_on_error:
                print("已中断。修复后可加 --from {} 从该阶段续跑。".format(name))
                return result.returncode
        else:
            print(f"[完成] {label}")

    print("\n" + "=" * 68)
    print(f"全流程完成，用时 {time.time() - started:.1f} 秒")
    print(f"可视化结果目录：{cfg.visualization_dir}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
