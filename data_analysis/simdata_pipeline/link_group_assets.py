# -*- coding: utf-8 -*-
"""
把「20组模拟数据」的每个子文件夹，回链到 batch_grouping 的分组产物
====================================================================

背景
----
`build_20_groups.py` 生成 20组模拟数据时，只把**原始腔室数据**搬了过来，
没有带上 batch_grouping 产出的分组清单与生长曲线页。本脚本补齐这一环：

    20组模拟数据/<自变量组合>/<…>_可视化结果.xlsx
        └─ 「数据来源」页签记录每个腔室取自哪个原始文件/页签
                │  归一化成 (日期_文件名, 原腔室名) 集合
                ▼
        数据汇总分析结果/<来源工况>/18_batch_*/selected__groups_<i>_<n>.xlsx
                │  组序号 i ≡ selected_groups_growth_curves_*.pptx 第 i 页
                ▼
        ① 复制该 xlsx 到子文件夹
        ② 新建 <来源工况>.pptx，只保留对应页

注意：**子文件夹名（自变量组合）通常 ≠ 来源工况**（μmax 是跨条件挑选后
重新配对的），所以必须全库搜索，不能只在同名工况目录下找。

用法
----
    python link_group_assets.py                  # 默认策略：已存在则跳过，不覆盖
    python link_group_assets.py --dry-run        # 只核对，不复制不建 pptx
    python link_group_assets.py --on-exist backup     # 已存在先备份再写入
    python link_group_assets.py --on-exist overwrite  # 直接覆盖
    python link_group_assets.py --agg-dir ... --out-dir ...

源目录（数据汇总分析结果）全程只读；所有写入都发生在「20组模拟数据」下。
"""
import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core import pptx_tools
from core.group_index import (chambers_of_output, load_group_index,
                              match_groups, match_groups_by_mu,
                              output_signature_fallback)
from core.logging_setup import setup_logger

POLICIES = ("skip", "backup", "overwrite")


# --------------------------------------------------------------------------- 覆盖策略
def apply_policy(target, policy, logger, dry_run=False):
    """
    返回 True 表示可以继续写入 target，False 表示应跳过。
    skip      —— 记录并跳过（默认，绝不覆盖）
    backup    —— 把已有文件改名成 *_backup_<时间戳> 后写入
    overwrite —— 直接覆盖
    """
    if not target.exists():
        return True
    if policy == "overwrite":
        logger.warning("  已存在，按策略覆盖：%s", target.name)
        return True
    if policy == "backup":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = target.with_name(f"{target.stem}_backup_{stamp}{target.suffix}")
        if dry_run:
            logger.info("  [dry-run] 已存在，将备份为 %s", backup.name)
            return True
        shutil.move(str(target), str(backup))
        logger.warning("  已存在 → 备份为 %s", backup.name)
        return True
    logger.warning("  已存在，跳过（未覆盖）：%s", target.name)
    return False


def find_curve_pptx(batch_dir, logger=None, cache=None):
    """定位 batch 目录下的生长曲线 pptx（多个取排序后第一个）。"""
    if cache is not None and batch_dir in cache:
        return cache[batch_dir]
    found = sorted(batch_dir.glob(config.CURVE_PPTX_PATTERN))
    pptx = found[0] if found else None
    if pptx is None and logger:
        logger.warning("目录下找不到 %s：%s", config.CURVE_PPTX_PATTERN, batch_dir)
    elif len(found) > 1 and logger:
        logger.warning("目录下有多个 %s，取 %s", config.CURVE_PPTX_PATTERN, pptx.name)
    if cache is not None:
        cache[batch_dir] = pptx
    return pptx


# --------------------------------------------------------------------------- 匹配
def match_one(xlsx, entries, logger, tag):
    """单个输出工作簿 → (匹配的 GroupEntry 列表, 匹配方式)。"""
    chambers = chambers_of_output(xlsx)
    if chambers:
        hits = match_groups(frozenset(chambers), entries)
        if hits:
            return hits, "腔室集合（数据来源页签）"
        logger.warning("%s 腔室集合未命中（%d 个腔室），改用 μmax 回退匹配",
                       tag, len(chambers))
    else:
        logger.warning("%s 缺少「数据来源」页签，改用 μmax 回退匹配", tag)

    sig = output_signature_fallback(xlsx)
    hits = match_groups_by_mu(sig, entries, config.MU_MATCH_TOL)
    if hits:
        return hits, f"μmax 回退（n={sig[0]}, μmax={sig[1]:.4f}）"
    return [], "未匹配"


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="20组模拟数据 ↔ 分组产物 回链")
    ap.add_argument("--agg-dir", default=str(config.AGG_RESULT_DIR),
                    help="汇总结果目录（数据汇总分析结果，只读）")
    ap.add_argument("--out-dir", default=str(config.OUTPUT_DIR),
                    help="目标目录（20组模拟数据）")
    ap.add_argument("--batch-dir", default=config.BATCH_DIR_NAME,
                    help="batch 子目录名，默认取 config.BATCH_DIR_NAME")
    ap.add_argument("--on-exist", default=config.LINK_ON_EXIST, choices=POLICIES,
                    help="目标文件已存在时的策略（默认 skip：不覆盖）")
    ap.add_argument("--dry-run", action="store_true", help="只核对，不复制不建 pptx")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    logger = setup_logger(Path(__file__).resolve().parent / "_logs")

    logger.info("=" * 74)
    logger.info("20组模拟数据 → 分组产物 回链")
    logger.info("  汇总结果（只读）: %s", args.agg_dir)
    logger.info("  目标目录        : %s", out_root)
    logger.info("  batch 子目录    : %s", args.batch_dir)
    logger.info("  已存在策略      : %s", args.on_exist)
    logger.info("  模式            : %s", "核对（不写入）" if args.dry_run else "写入")

    entries = load_group_index(args.agg_dir, logger=logger, batch_name=args.batch_dir)
    if not entries:
        logger.error("未扫到任何 selected__groups_*.xlsx，终止")
        return 1

    pptx_cache = {}
    subfolders = sorted(p for p in out_root.iterdir() if p.is_dir())
    logger.info("待处理子文件夹：%d 个", len(subfolders))

    report = []        # 每个子文件夹的结果
    unmatched = []     # 未匹配项
    skipped_exist = []

    logger.info("-" * 74)
    for folder in subfolders:
        # 只处理本管线产出的数据表格；排除已复制进来的分组文件与 Excel 临时文件
        candidates = sorted(p for p in folder.glob("*.xlsx")
                            if not p.name.startswith("~$")
                            and not p.name.startswith("selected__groups_"))
        if not candidates:
            logger.warning("[%s] 子文件夹内没有数据表格，跳过", folder.name)
            report.append({"folder": folder.name, "matched": [], "unmatched": [],
                           "xlsx": [], "pptx": []})
            continue

        matched = []       # [(输出文件, entry, 匹配方式)]
        for xlsx in candidates:
            tag = f"[{folder.name}/{xlsx.name}]"
            hits, how = match_one(xlsx, entries, logger, tag)
            if not hits:
                logger.warning("%s 未能匹配任何数据组，跳过并记录", tag)
                unmatched.append(f"{folder.name}/{xlsx.name}")
                continue
            logger.info("%s 命中 %d 个数据组（%s）：%s", tag, len(hits), how,
                        ", ".join(f"{e.condition} 组{e.index}(n={e.size})"
                                  for e in hits))
            for e in hits:
                matched.append((xlsx, e, how))

        # ---------- ① 复制 xlsx ----------
        copied_xlsx = []
        used_names = {}
        for _xlsx, e, _how in matched:
            name = e.path.name
            prev_cond = used_names.get(name)
            if prev_cond is not None and prev_cond != e.condition:
                # 不同工况的同名分组文件落到同一子文件夹 → 加工况前缀区分
                name = f"{e.condition}__{name}"
                logger.warning("[%s] 同名分组文件冲突，改用 %s", folder.name, name)
            used_names[name] = e.condition
            target = folder / name
            if not apply_policy(target, args.on_exist, logger, args.dry_run):
                skipped_exist.append(str(target))
                continue
            if not args.dry_run:
                shutil.copy2(str(e.path), str(target))     # 源目录只读，只做复制
            copied_xlsx.append((name, e))
            logger.info("  [%s] %s复制 xlsx：%s ← %s/组%d",
                        folder.name, "[dry-run] 将" if args.dry_run else "",
                        name, e.condition, e.index)

        # ---------- ②③ 按来源工况建 pptx ----------
        by_cond = defaultdict(list)
        for _xlsx, e, _how in matched:
            by_cond[e.condition].append(e)

        made_pptx = []
        for cond in sorted(by_cond):
            cond_entries = sorted(by_cond[cond], key=lambda x: x.index)
            pages = [e.index for e in cond_entries]
            src_pptx = find_curve_pptx(cond_entries[0].batch_dir, logger, pptx_cache)
            if src_pptx is None:
                logger.warning("[%s] 工况「%s」无生长曲线 pptx，跳过建 pptx",
                               folder.name, cond)
                continue

            # 页标题交叉校验：确认「第 i 页」确实是该组，防止页序与文件错位
            titles = dict(pptx_tools.slide_titles(src_pptx))
            for e in cond_entries:
                problem = pptx_tools.check_title(e.index, titles.get(e.index),
                                                 e.index, e.size)
                if problem:
                    logger.warning("[%s] %s", folder.name, problem)

            pptx_name = config.PPTX_NAME_TEMPLATE.format(condition=cond)
            target = folder / pptx_name
            if not apply_policy(target, args.on_exist, logger, args.dry_run):
                skipped_exist.append(str(target))
                continue
            if not args.dry_run:
                keep, missing, total = pptx_tools.extract_slides(src_pptx, pages, target)
            else:
                total = len(pptx_tools.slide_titles(src_pptx))
                keep = [p for p in pages if 1 <= p <= total]
                missing = [p for p in pages if not (1 <= p <= total)]
            if missing:
                logger.warning("[%s] 组序号 %s 超出 pptx 页数（%d 页），已跳过",
                               folder.name, missing, total)
            if keep:
                made_pptx.append((pptx_name, cond, keep))
                logger.info("  [%s] %s新建 pptx：%s（源 %d 页 → 保留第 %s 页）",
                            folder.name, "[dry-run] 将" if args.dry_run else "",
                            pptx_name, total, "/".join(str(p) for p in keep))
            elif not args.dry_run and target.exists():
                target.unlink()          # 没有可用页，删掉刚生成的空演示
                logger.warning("[%s] 无可保留页，已删除空的 %s", folder.name, pptx_name)

        report.append({
            "folder": folder.name,
            "matched": [(e.condition, e.index, e.size) for _x, e, _h in matched],
            "unmatched": [x.name for x in candidates
                          if f"{folder.name}/{x.name}" in unmatched],
            "xlsx": copied_xlsx,
            "pptx": made_pptx,
        })

    # ------------------------------------------------------------------ 汇总日志
    logger.info("=" * 74)
    logger.info("汇总报告")
    logger.info("-" * 74)
    n_xlsx = n_pptx = 0
    for item in report:
        logger.info("■ %s", item["folder"])
        if item["matched"]:
            for cond, idx, size in item["matched"]:
                logger.info("    匹配：%s 组%d（n=%d）", cond, idx, size)
        for name, e in item["xlsx"]:
            logger.info("    复制 xlsx：%s  ← %s 组%d", name, e.condition, e.index)
            n_xlsx += 1
        for name, cond, pages in item["pptx"]:
            logger.info("    新建 pptx：%s（工况 %s，第 %s 页）",
                        name, cond, "/".join(str(p) for p in pages))
            n_pptx += 1
        if not item["matched"]:
            logger.info("    （无匹配）")

    logger.info("-" * 74)
    logger.info("未匹配项（%d）：%s",
                len(unmatched), "、".join(unmatched) if unmatched else "无")
    logger.info("因已存在而跳过（%d）：%s",
                len(skipped_exist),
                "、".join(Path(p).name for p in skipped_exist) if skipped_exist else "无")
    logger.info("合计：复制 xlsx %d 个，新建 pptx %d 个（源目录未做任何改动）",
                n_xlsx, n_pptx)
    logger.info("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
