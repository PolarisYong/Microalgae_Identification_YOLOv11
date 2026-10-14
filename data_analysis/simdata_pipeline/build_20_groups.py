# -*- coding: utf-8 -*-
"""
构建「20组模拟数据」
==================
流程：
    CCF 映射表（20 行）
        ├─ 自变量组合（光照/氨氮/无机碳）→ 输出文件夹名
        └─ 原始数据来源 → 来源工况 + Response 序号
              └─ 数据汇总分析结果/<来源工况>/18_batch_*/selected__groups_<i>_<n>.xlsx
                    └─ 已选腔室列表 → μmax_revision_原始数据/…/CH<k>_可视化结果_<N>.xlsx
                          └─ 页签 CH<k>_CB<j> → 重命名为 CH1_CB<序号> 写入输出

用法：
    python build_20_groups.py                 # 用 config.py 里的默认路径
    python build_20_groups.py --dry-run       # 只做映射核对，不写文件
    python build_20_groups.py --raw-dir ... --out-dir ...

异常策略：缺失文件 / 腔室数不等于 18 → 记录 WARNING 并继续，不中断。
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (AGG_RESULT_DIR, CCF_MAPPING_XLSX, CULTURE_HOURS,
                    EXPECTED_CHAMBERS, OUTPUT_DIR, PLOT_DPI, RAW_DATA_DIR, PROJECT_ROOT)
from core.ccf_table import load_ccf_rows
from core.conditions import code_to_name, list_condition_dirs
from core.groups import load_groups, pick_group
from core import rawdata
from core.logging_setup import setup_logger
from core.writer import build_workbook


def main():
    ap = argparse.ArgumentParser(description="构建 CCF 20 组模拟数据")
    ap.add_argument("--ccf-xlsx", default=str(CCF_MAPPING_XLSX))
    ap.add_argument("--agg-dir", default=str(AGG_RESULT_DIR))
    ap.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--hours", type=int, default=CULTURE_HOURS)
    ap.add_argument("--dry-run", action="store_true", help="只核对映射，不写文件")
    ap.add_argument("--no-plots", action="store_true",
                    help="不嵌入可视化图（只写数据表，速度更快、体积更小）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    # 日志放在项目侧，避免污染输出目录
    logger = setup_logger(Path(__file__).resolve().parent / "_logs")

    logger.info("=" * 72)
    logger.info("构建 20 组模拟数据")
    logger.info("  CCF 映射表 : %s", args.ccf_xlsx)
    logger.info("  聚合结果   : %s", args.agg_dir)
    logger.info("  原始数据   : %s", args.raw_dir)
    logger.info("  输出目录   : %s", out_dir)
    logger.info("  培养时长   : %s 小时", args.hours)
    logger.info("  可视化图   : %s（dpi=%d）",
                "关闭" if args.no_plots else "开启", PLOT_DPI)
    logger.info("  模式       : %s", "核对（不写文件）" if args.dry_run else "写出")

    # ---------- 1. 工况目录 ----------
    cond_map, skipped = list_condition_dirs(args.agg_dir)
    if skipped:
        logger.warning("以下目录名无法解析为工况，已忽略：%s", skipped)
    logger.info("扫描到 %d 个工况目录", len(cond_map))

    # ---------- 2. CCF 20 行 ----------
    rows = load_ccf_rows(args.ccf_xlsx, logger=logger)

    # 自变量组合出现次数（中心点会重复 6 次）→ 决定文件名是否加序号
    dup_counter = Counter(r["design_code"] for r in rows)
    seen = defaultdict(int)

    logger.info("-" * 72)
    ok_count = 0
    warn_count = 0

    for r in rows:
        tag = f"[行{r['row']}]"
        design_code = r["design_code"]
        folder_name = code_to_name(*design_code)
        src = r["source"]

        seen[design_code] += 1
        n_total = dup_counter[design_code]
        suffix = "" if n_total == 1 else f"_{seen[design_code]:02d}"
        base_name = f"{folder_name}_可视化结果{suffix}"

        logger.info("%s 自变量组合 %s → 文件夹「%s」 | μmax=%.4f | 来源=%s",
                    tag, design_code, folder_name, r["mu_max"] or 0, r["source_raw"])

        # ---------- 3. 定位来源工况 ----------
        if src is None or src["code"] is None:
            logger.warning("%s 「原始数据来源」无法解析：%s，跳过该组", tag, r["source_raw"])
            warn_count += 1
            continue
        src_code = src["code"]
        cond = cond_map.get(src_code)
        if cond is None:
            logger.warning("%s 来源工况 %s（%s）在数据汇总分析结果中不存在，跳过该组",
                           tag, src_code, code_to_name(*src_code))
            warn_count += 1
            continue

        # ---------- 4. 取已选分组 ----------
        groups = load_groups(cond["path"], logger=logger)
        grp, kind = pick_group(groups, r["mu_max"], src.get("res"),
                               logger=logger, tag=tag)
        if grp is None:
            logger.warning("%s 未能在工况「%s」中定位到分组，跳过该组", tag, cond["name"])
            warn_count += 1
            continue
        logger.info("%s 采用第 %d 组（%d 个腔室，组均值 %.4f，定位方式：%s）→ 来源工况「%s」",
                    tag, grp["index"], len(grp["members"]), grp["mean_mu"],
                    kind, cond["name"])

        # ---------- 5. 逐个腔室取原始数据 ----------
        chambers = []
        miss_file, miss_sheet = 0, 0
        for i, (src_file, src_sheet, _mu) in enumerate(grp["members"], start=1):
            raw_path = rawdata.resolve_raw_path(src_file, raw_dir=args.raw_dir,
                                                hours=args.hours)
            if raw_path is None:
                logger.warning("%s 找不到原始数据文件：%s（腔室 %s）",
                               tag, src_file, src_sheet)
                miss_file += 1
                continue
            rows_data, params = rawdata.read_chamber_sheet(raw_path, src_sheet,
                                                           logger=logger, tag=tag)
            if rows_data is None:
                miss_sheet += 1
                continue
            try:
                rel = raw_path.relative_to(Path(args.raw_dir))
            except ValueError:
                rel = raw_path
            chambers.append({
                "new_name": f"CH1_CB{i}",
                "rows": rows_data,
                "counts": rawdata.extract_counts(rows_data, args.hours + 1),
                "areas": rawdata.extract_areas(rows_data, args.hours + 1),
                "params": params,
                "src_file": src_file,
                "src_sheet": src_sheet,
                "src_rel": str(rel),
            })

        if miss_file or miss_sheet:
            logger.warning("%s 有 %d 个腔室缺文件、%d 个腔室缺页签，已跳过",
                           tag, miss_file, miss_sheet)
            warn_count += 1
        if len(chambers) != EXPECTED_CHAMBERS:
            logger.warning("%s 腔室数 %d ≠ 期望 %d（该组实际规模如此）",
                           tag, len(chambers), EXPECTED_CHAMBERS)
            warn_count += 1
        if not chambers:
            logger.warning("%s 无任何可用腔室，跳过写出", tag)
            continue

        # ---------- 6. 写出 ----------
        out_path = out_dir / folder_name / f"{base_name}.xlsx"
        if args.dry_run:
            logger.info("%s [dry-run] 将写出 %s", tag, out_path)
        else:
            build_workbook(out_path, chambers, logger=logger, tag=tag,
                           condition=folder_name, plots=not args.no_plots)
        ok_count += 1

    rawdata.close_cache()

    logger.info("-" * 72)
    logger.info("完成：成功 %d 组，告警 %d 处", ok_count, warn_count)
    logger.info("输出目录：%s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
