# -*- coding: utf-8 -*-
"""
腔室数据二次分组
================
背景
----
`数据汇总分析结果/<自变量文件夹>/18_batch_*/<源>_marked.xlsx` 的「汇总结果」页签里，
标红行 = 该腔室已被 batch_grouping 选进某个数据组（即已用于「20组模拟数据」与
「未使用数据组」）。本脚本把**其余尚未参与分组的腔室**挑出来，做二次分组。

流程
----
    ① 扫描     遍历自变量文件夹 → 用标红行剔除已使用腔室 → 得到未参与分组的腔室
    ② 汇总     每个腔室一个页签写入「未参与数据分组腔室数据.xlsx」
               （每份 ≤ 50 个页签，超出拆 _part2/_part3…；H/I 列标注实际来源）
    ③ 筛查     按微藻生长规律剔除异常腔室 → 「剔除腔室数据.xlsx」（附剔除原因）
    ④ 分组     剩余腔室按 batch_grouping 的方式分组（CV<15% 且 JB p>0.05，
               目标 18 个/组、最少 12 个/组），输出到「所有未参与数据分组腔室数据」
    ⑤ 归档     「文档数据来源.xlsx」逐条记录每组成员的实际来源

输出目录
--------
    所有未参与数据分组腔室数据/
        ├─ 未参与数据分组腔室数据.xlsx（+ _part2…）
        ├─ 剔除腔室数据.xlsx（+ _part2…）
        ├─ 文档数据来源.xlsx
        └─ <μmax 编码>/
                ├─ <μmax 编码>_可视化结果.xlsx
                ├─ selected__groups_<i>_<n>.xlsx
                └─ <μmax 编码>.pptx

用法
----
    python build_second_round.py                  # 正式写出
    python build_second_round.py --dry-run        # 只核对，不写任何文件
    python build_second_round.py --no-plots       # 不嵌图（快很多）
    python build_second_round.py --per-condition  # 改为「每个自变量文件夹各自分组」
    python build_second_round.py --force          # 已存在的组文件夹重算覆盖

源目录（数据汇总分析结果）全程只读。
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core import group_assets, screening, second_round, unused_scan
from core.logging_setup import setup_logger


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description="未参与分组腔室的二次分组")
    ap.add_argument("--agg-dir", default=str(config.AGG_RESULT_DIR),
                    help="汇总结果目录（数据汇总分析结果，只读）")
    ap.add_argument("--out-dir", default=str(config.SECOND_ROUND_DIR),
                    help="输出根目录（所有未参与数据分组腔室数据）")
    ap.add_argument("--batch-dir", default=config.BATCH_DIR_NAME,
                    help="batch 子目录名（存放 _marked.xlsx）")
    ap.add_argument("--min-size", type=int, default=config.GROUP_MIN_SIZE)
    ap.add_argument("--max-size", type=int, default=config.GROUP_MAX_SIZE)
    ap.add_argument("--cv", type=float, default=config.GROUP_CV_LIMIT)
    ap.add_argument("--p", type=float, default=config.GROUP_P_LIMIT)
    ap.add_argument("--per-condition", action="store_true",
                    help="按自变量文件夹各自分组（默认把所有未使用腔室视为同一自变量下的数据）")
    ap.add_argument("--no-plots", action="store_true", help="不嵌可视化图")
    ap.add_argument("--force", action="store_true", help="已存在的数据组文件夹重算覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只核对，不写任何文件")
    args = ap.parse_args()

    logger = setup_logger(Path(__file__).resolve().parent / "_logs",
                          name="build_second_round")

    agg_dir = Path(args.agg_dir)
    out_dir = Path(args.out_dir)

    logger.info("=" * 78)
    logger.info("未参与分组腔室 —— 二次分组")
    logger.info("  汇总结果（只读）: %s", agg_dir)
    logger.info("  输出目录        : %s", out_dir)
    logger.info("  分组参数        : 目标 %d 个/组，最少 %d 个/组，CV < %.2f，JB p > %.3f",
                args.max_size, args.min_size, args.cv, args.p)
    logger.info("  分组范围        : %s", "按自变量文件夹各自分组" if args.per_condition
                else "全部未使用腔室视为同一自变量下的数据（单一池）")
    logger.info("  可视化图        : %s", "关闭" if args.no_plots else "开启")
    logger.info("  模式            : %s", "核对（不写入）" if args.dry_run else "写入")

    if not agg_dir.is_dir():
        logger.error("汇总结果目录不存在：%s", agg_dir)
        return 1
    if args.min_size < 2 or args.max_size <= args.min_size or args.max_size > 18:
        logger.error("分组规模必须满足 2 <= 最小规模 < 目标规模 <= 18")
        return 1
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- ① 扫描
    logger.info("-" * 78)
    scan = unused_scan.scan_unused(agg_dir, logger=logger, batch_name=args.batch_dir)
    refs = scan["refs"]
    if not refs:
        logger.error("没有找到任何未参与分组的腔室，终止")
        return 1
    if scan["no_marked"]:
        logger.warning("以下源文件找不到 _marked.xlsx，其全部腔室按「未使用」处理（%d 个）：",
                       len(scan["no_marked"]))
        for s in scan["no_marked"]:
            logger.warning("  · %s", s)

    # ---------------------------------------------------------------- ② 汇总全部未使用腔室
    logger.info("-" * 78)
    if args.dry_run:
        parts = max(1, (len(refs) + config.MAX_SHEETS_PER_FILE - 1)
                    // config.MAX_SHEETS_PER_FILE)
        logger.info("[dry-run] 将写出「%s」共 %d 个腔室页签 → 拆成 %d 个文件",
                    config.UNUSED_CHAMBER_XLSX, len(refs), parts)
    else:
        second_round.write_chamber_workbooks(
            refs, out_dir, config.UNUSED_CHAMBER_XLSX[:-5],
            logger=logger, tag="[汇总]")

    # ---------------------------------------------------------------- ③ 生长规律筛查
    logger.info("-" * 78)
    kept, excluded = screening.partition(refs, logger=logger, tag="[筛查]")
    reasons = {ref.uid: reason for ref, reason in excluded}

    by_reason = collections.Counter(reason.split("：")[0] for _, reason in excluded)
    for reason, count in by_reason.most_common():
        logger.info("  剔除原因 %-16s %d 个", reason, count)

    if args.dry_run:
        logger.info("[dry-run] 将写出「%s」：%d 个腔室",
                    config.EXCLUDED_CHAMBER_XLSX, len(excluded))
    elif excluded:
        second_round.write_chamber_workbooks(
            [ref for ref, _ in excluded], out_dir, config.EXCLUDED_CHAMBER_XLSX[:-5],
            logger=logger, tag="[剔除]", reasons=reasons)

    # ---------------------------------------------------------------- ④ 分组
    logger.info("-" * 78)
    strict = [r for r in kept
              if r.n0 > config.STRICT_MIN_N0 and r.r2 > config.STRICT_MIN_R2]
    logger.info("筛查后保留 %d 个 → batch_grouping 严格候选（N0 > 0 且 R² > %.2f）%d 个"
                "｜因拟合质量未参与分组 %d 个（仍保留在汇总文件中）",
                len(kept), config.STRICT_MIN_R2, len(strict), len(kept) - len(strict))
    if len(strict) < args.min_size:
        logger.error("严格候选仅 %d 个，少于最小组规模 %d，无法分组",
                     len(strict), args.min_size)
        return 1

    refs_by_key = {}
    for r in refs:
        refs_by_key[(f"{r.condition}/{r.src_file}", r.sheet)] = r

    pools = []
    if args.per_condition:
        by_cond = collections.defaultdict(list)
        for r in strict:
            by_cond[r.condition].append(r)
        for cond in sorted(by_cond):
            pools.append((cond, by_cond[cond]))
    else:
        pools.append(("全部", strict))

    all_groups, taken = [], set()
    pool_strict = {}
    work_dir = out_dir / "_work"
    for pool_name, pool_refs in pools:
        if len(pool_refs) < args.min_size:
            logger.warning("分组池「%s」只有 %d 个腔室，少于最小组规模 %d，跳过",
                           pool_name, len(pool_refs), args.min_size)
            continue
        rows = second_round.to_group_rows(pool_refs)
        results, groups, remaining, full_count, metrics = group_assets.allocate(
            rows, min_size=args.min_size, max_size=args.max_size,
            cv_limit=args.cv, p_limit=args.p)
        pool_strict[pool_name] = results        # 全量候选，供 strict_<N> 页签使用
        logger.info("分组池「%s」：候选 %d 个 → 成型 %d 组（%s）｜未成组 %d 个",
                    pool_name, len(pool_refs), len(groups),
                    "、".join(str(len(g)) for g in groups) or "无", len(remaining))
        if not groups:
            continue

        # 按 μmax 均值升序命名（与「未使用数据组」一致）
        scored = sorted(((metrics(g).mean, i, g) for i, g in enumerate(groups)),
                        key=lambda x: (x[0], x[1]))
        for rank, (mean, _i, group) in enumerate(scored, start=1):
            base = second_round.mu_folder_name(mean)
            name, k = base, 2
            while name in taken:
                name = f"{base}_{k:02d}"
                k += 1
            taken.add(name)
            all_groups.append((rank, name, mean, group, pool_name))

    if not all_groups:
        logger.error("没有任何数据组成型，终止")
        return 1

    logger.info("-" * 78)
    built = []
    for rank, name, mean, group, pool_name in all_groups:
        folder = out_dir / name
        tag = f"[{rank}/{len(all_groups)}] {name}"
        stats = metrics(group)
        logger.info("%s %s｜n=%d｜μmax 均值 %.6f h⁻¹｜CV %.2f%%｜JB p %.3f",
                    tag, pool_name, len(group), stats.mean,
                    stats.cv * 100, stats.p_value)

        if not args.dry_run and folder.is_dir() and not args.force:
            if (folder / f"{name}{config.GENERATED_SUFFIX}").is_file():
                logger.info("%s 已存在，跳过（需要重算请加 --force）", tag)
                built.append((name, rank, group))
                continue

        if args.dry_run:
            logger.info("%s [dry-run] 将生成：%s_可视化结果.xlsx / "
                        "selected__groups_%d_%d.xlsx / %s",
                        tag, name, rank, len(group),
                        config.SECOND_PPTX_NAME_TEMPLATE.format(folder=name))
            built.append((name, rank, group))
            continue

        res = group_assets.build_group_assets(
            rank, group, refs_by_key, folder, work_dir,
            logger=logger, tag=tag, plots=not args.no_plots,
            strict_rows=pool_strict.get(pool_name))
        built.append((name, rank, group))
        if res["notes"]:
            for note in res["notes"]:
                logger.warning("%s %s", tag, note)
        logger.info("%s 产物：%s", tag,
                    "、".join(x for x in (res["xlsx"], res["selected"], res["pptx"]) if x)
                    or "无")

    # ---------------------------------------------------------------- ⑤ 文档数据来源
    logger.info("-" * 78)
    if args.dry_run:
        logger.info("[dry-run] 将写出「%s」：%d 个数据组 / %d 条成员记录",
                    config.DOC_SOURCE_XLSX, len(built),
                    sum(len(g) for _, _, g in built))
    else:
        strict_keys = {r.uid for r in strict}
        second_round.write_doc_source(
            out_dir / config.DOC_SOURCE_XLSX,
            [(name, rank, group, refs_by_key) for name, rank, group in built],
            excluded, refs, strict_keys, logger=logger)

    # ---------------------------------------------------------------- 统计
    logger.info("=" * 78)
    logger.info("统计")
    logger.info("  自变量文件夹数        : %d", scan["conditions"])
    logger.info("  读取源工作簿数        : %d", scan["files"])
    logger.info("  已使用腔室（标红）    : %d", scan["used"])
    logger.info("  未参与分组腔室        : %d", len(refs))
    logger.info("  生长规律剔除          : %d", len(excluded))
    logger.info("  筛查后保留            : %d", len(kept))
    logger.info("  严格候选（参与分组）  : %d", len(strict))
    logger.info("  生成数据组            : %d（%s）", len(built),
                "、".join(f"{name}(n={len(g)})" for name, _, g in built))
    if scan["missing_sheet"]:
        logger.info("  缺页签被跳过          : %d", len(scan["missing_sheet"]))
    logger.info("输出目录：%s", out_dir)
    logger.info("完成（源目录未做任何改动）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
