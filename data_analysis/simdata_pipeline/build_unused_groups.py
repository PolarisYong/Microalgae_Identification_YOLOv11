# -*- coding: utf-8 -*-
"""
构建「未使用数据」
================
背景
----
`数据汇总分析结果/<工况>/18_batch_*/selected__groups_*.xlsx` 共有 40 多个数据组，
CCF 响应面只用了其中 20 个（体现在「20组模拟数据」里）。本脚本把**剩下未使用的数据组**
逐个导出，方便后续核查/补做分析。

判定「已使用」
--------------
20组模拟数据 每个子文件夹里的工作簿都带「数据来源」页签，记录每个腔室取自
哪个原始文件/页签。把它归一化成 (日期_文件名, 原腔室名) 集合，与分组文件做
**集合相等**比对，命中的分组即视为已使用（复用 link_group_assets 的匹配逻辑）。

处理内容（每个未使用数据组一个子文件夹）
----------------------------------------
    未使用数据/<μmax 编码>/
        ├─ <μmax 编码>_可视化结果.xlsx    # 与 20组模拟数据 同款：汇总(三张表+3图)
        │                                 #   + CH1_CB1..CBn（各 4 图）+ 数据来源
        ├─ selected__groups_<i>_<n>.xlsx  # 从源目录复制（源目录只读）
        └─ <所属工况>.pptx                # 从 growth_curves pptx 抽第 i 页另存

子文件夹命名：读分组文件 summary 页签 A 列的 `mu_max_mean_h^-1`，取 B 列数值，
去掉前导「0.」后取前 4 位（0.0657777777777778 → "0657"）；重名自动加 _02/_03 后缀。

幂等性
------
重复运行不会重复建文件夹：启动时先扫描已存在的子文件夹，用其工作簿的
「数据来源」页签反查所属数据组，已处理过的直接跳过；上次中途失败留下的
空/残缺文件夹会被复用而不是另建。

用法
----
    python build_unused_groups.py                 # 正式写出（已存在默认跳过）
    python build_unused_groups.py --dry-run       # 只核对，不写任何文件
    python build_unused_groups.py --no-plots      # 不嵌图（快很多）
    python build_unused_groups.py --on-exist backup|overwrite

源目录（数据汇总分析结果 / μmax_revision_原始数据）全程只读，只做复制与读取。
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core import pptx_tools, rawdata
from core.group_index import (chambers_of_output, load_group_index, match_groups,
                              match_groups_by_mu, output_signature_fallback,
                              read_summary_metric)
from core.groups import read_members
from core.logging_setup import setup_logger
from core.writer import build_workbook
from link_group_assets import POLICIES, apply_policy, find_curve_pptx

# 本脚本自己产出的表格后缀，用于把「生成的表格」与「复制进来的分组文件」区分开。
# 中心点重复组会带序号后缀（…_可视化结果_01.xlsx），故用「包含」而不是「结尾」匹配。
GENERATED_SUFFIX = "_可视化结果.xlsx"
GENERATED_PATTERN = "*可视化结果*.xlsx"


def is_generated(path):
    """是否是本管线生成的数据表格（排除复制进来的 selected__groups_*.xlsx）。"""
    name = path.name
    return (not name.startswith("~$")
            and not name.startswith("selected__groups_")
            and "可视化结果" in name)


# --------------------------------------------------------------------------- 命名
def mu_folder_name(value, digits=None):
    """
    μmax 均值 → 文件夹名：去掉前导「0.」后取前 digits 位。

        0.0657777777777778 → "0657"
        0.24               → "2400"

    用 %.10f 固定展开，避免科学计数法或浮点尾数导致截断出错。
    """
    digits = digits or config.UNUSED_FOLDER_DIGITS
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{abs(v):.10f}"                       # 例：0.0657777778
    if "." in text:
        int_part, frac = text.split(".", 1)
    else:
        int_part, frac = text, ""
    digits_part = (int_part + frac) if int_part not in ("0", "") else frac
    return digits_part[:digits] or "0000"


# --------------------------------------------------------------------------- 已使用判定
def collect_used(entries, used_dir, logger):
    """
    扫描「20组模拟数据」下所有生成的工作簿 → 已使用数据组的 {(工况, 组序号)}。
    """
    used = set()
    used_dir = Path(used_dir)
    if not used_dir.is_dir():
        logger.warning("目录不存在，无法判定已使用的数据组：%s", used_dir)
        return used

    for xlsx in sorted(p for p in used_dir.rglob(GENERATED_PATTERN) if is_generated(p)):
        tag = f"[{xlsx.parent.name}/{xlsx.name}]"
        keys = chambers_of_output(xlsx)
        hits = []
        how = ""
        if keys:
            hits = match_groups(frozenset(keys), entries)
            how = "腔室集合"
        if not hits:
            sig = output_signature_fallback(xlsx)
            hits = match_groups_by_mu(sig, entries, config.MU_MATCH_TOL)
            how = "μmax 回退"
        for e in hits:
            used.add((e.condition, e.index))
        if not hits:
            logger.warning("%s %s 未匹配到任何数据组（不影响未使用判定）",
                           tag, xlsx.name)
        else:
            logger.info("%s %s → 已使用：%s（%s）", tag, xlsx.name,
                        "、".join(f"{e.condition} 组{e.index}" for e in hits), how)
    logger.info("已使用数据组：%d 个", len(used))
    return used


# --------------------------------------------------------------------------- 已存在的子文件夹
def scan_existing(unused_root, entries, logger):
    """
    扫描「未使用数据」下已存在的子文件夹。

    返回 (processed, orphans)
        processed : {(工况, 组序号): 文件夹路径}   已经完整处理过的
        orphans   : {文件夹名: 文件夹路径}         上次中途失败留下的残缺文件夹
    """
    processed, orphans = {}, {}
    if not unused_root.is_dir():
        return processed, orphans

    for folder in sorted(p for p in unused_root.iterdir() if p.is_dir()):
        generated = sorted(p for p in folder.glob(GENERATED_PATTERN) if is_generated(p))
        matched_key = None
        for xlsx in generated:
            keys = chambers_of_output(xlsx)
            hits = match_groups(frozenset(keys), entries) if keys else []
            if len(hits) == 1:
                matched_key = (hits[0].condition, hits[0].index)
                break
        if matched_key:
            processed[matched_key] = folder
        else:
            orphans[folder.name] = folder
            logger.warning("发现残缺文件夹（无有效数据表格，将被复用或保留）：%s",
                           folder.name)
    return processed, orphans


# --------------------------------------------------------------------------- 单个数据组
def build_one(entry, folder, args, logger, pptx_cache):
    """
    为一个未使用数据组建立子文件夹内容。返回 dict（成功/失败与各产物清单）。
    """
    tag = f"[{entry.condition} 组{entry.index}]"
    result = {"ok": False, "xlsx": None, "copied": None, "pptx": None,
              "n_chambers": 0, "notes": []}

    # dry-run 下不建目录，保证「只核对、不写」的语义
    if not args.dry_run:
        folder.mkdir(parents=True, exist_ok=True)

    # ---------- ③ 生成数据表格（与 20组模拟数据 同款） ----------
    members = read_members(entry.path, logger=logger, tag=tag)
    if not members:
        result["notes"].append("分组文件里读不到已选腔室清单")
        return result

    chambers = []
    miss_file = miss_sheet = 0
    for i, (src_file, src_sheet, _mu) in enumerate(members, start=1):
        raw_path = rawdata.resolve_raw_path(src_file, raw_dir=args.raw_dir,
                                            hours=args.hours)
        if raw_path is None:
            logger.warning("%s 找不到原始数据文件：%s（腔室 %s）", tag, src_file, src_sheet)
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
        result["notes"].append(f"缺文件 {miss_file} 个 / 缺页签 {miss_sheet} 个")
        logger.warning("%s 缺文件 %d 个腔室、缺页签 %d 个腔室", tag, miss_file, miss_sheet)
    if not chambers:
        result["notes"].append("无任何可用腔室")
        return result
    if len(chambers) != entry.size:
        result["notes"].append(f"腔室数 {len(chambers)} ≠ 文件名标记 {entry.size}")

    result["n_chambers"] = len(chambers)
    out_xlsx = folder / f"{folder.name}{GENERATED_SUFFIX}"
    if args.dry_run:
        logger.info("%s [dry-run] 将写出 %s（%d 个腔室）",
                    tag, out_xlsx.name, len(chambers))
    else:
        if not apply_policy(out_xlsx, args.on_exist, logger, args.dry_run):
            result["notes"].append("数据表格已存在且策略为跳过")
        else:
            build_workbook(out_xlsx, chambers, logger=logger, tag=tag,
                           condition=entry.condition, plots=not args.no_plots)
            logger.info("%s 数据表格：%s（%d 个腔室）", tag, out_xlsx.name, len(chambers))
    result["xlsx"] = out_xlsx.name

    # ---------- ① 复制分组文件 ----------
    target = folder / entry.path.name
    if apply_policy(target, args.on_exist, logger, args.dry_run):
        if not args.dry_run:
            shutil.copy2(str(entry.path), str(target))    # 源目录只读，只复制
        logger.info("%s 复制分组文件：%s", tag, target.name)
    result["copied"] = target.name

    # ---------- ② 抽取对应幻灯片页 ----------
    src_pptx = find_curve_pptx(entry.batch_dir, logger, pptx_cache)
    if src_pptx is None:
        result["notes"].append("所属工况下找不到 growth_curves pptx")
        return result

    titles = dict(pptx_tools.slide_titles(src_pptx))
    problem = pptx_tools.check_title(entry.index, titles.get(entry.index),
                                     entry.index, entry.size)
    if problem:
        result["notes"].append(problem)
        logger.warning("%s %s", tag, problem)

    pptx_name = config.PPTX_NAME_TEMPLATE.format(condition=entry.condition)
    target = folder / pptx_name
    if not apply_policy(target, args.on_exist, logger, args.dry_run):
        result["notes"].append("pptx 已存在且策略为跳过")
    elif args.dry_run:
        total = len(titles)
        keep = [entry.index] if 1 <= entry.index <= total else []
        logger.info("%s [dry-run] 将新建 pptx：%s（第 %s 页）",
                    tag, pptx_name, "/".join(str(p) for p in keep))
        if keep:
            result["pptx"] = (pptx_name, keep)
        else:
            result["notes"].append(f"组序号 {entry.index} 超出 pptx 页数（{total} 页）")
    else:
        keep, missing, total = pptx_tools.extract_slides(src_pptx, [entry.index], target)
        if missing:
            result["notes"].append(f"组序号 {entry.index} 超出 pptx 页数（{total} 页）")
            logger.warning("%s 组序号 %s 超出 pptx 页数（%d 页）", tag, missing, total)
        if keep:
            result["pptx"] = (pptx_name, keep)
            logger.info("%s 新建 pptx：%s（源 %d 页 → 保留第 %s 页）",
                        tag, pptx_name, total, "/".join(str(p) for p in keep))
        elif target.exists():
            target.unlink()
            logger.warning("%s 无可保留页，已删除空的 %s", tag, pptx_name)

    result["ok"] = bool(result["xlsx"] and result["copied"] and result["pptx"])
    return result


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="导出未被 CCF 20 组使用的数据组")
    ap.add_argument("--agg-dir", default=str(config.AGG_RESULT_DIR),
                    help="汇总结果目录（数据汇总分析结果，只读）")
    ap.add_argument("--used-dir", default=str(config.OUTPUT_DIR),
                    help="已使用数据所在目录（20组模拟数据），用于判定哪些组已用")
    ap.add_argument("--unused-dir", default=str(config.UNUSED_DIR),
                    help="输出目录（未使用数据）")
    ap.add_argument("--raw-dir", default=str(config.RAW_DATA_DIR))
    ap.add_argument("--hours", type=int, default=config.CULTURE_HOURS)
    ap.add_argument("--batch-dir", default=config.BATCH_DIR_NAME)
    ap.add_argument("--on-exist", default=config.LINK_ON_EXIST, choices=POLICIES)
    ap.add_argument("--dry-run", action="store_true", help="只核对，不写文件")
    ap.add_argument("--no-plots", action="store_true", help="不嵌可视化图")
    args = ap.parse_args()

    logger = setup_logger(Path(__file__).resolve().parent / "_logs",
                          name="build_unused")

    logger.info("=" * 74)
    logger.info("导出未使用数据组")
    logger.info("  汇总结果（只读）: %s", args.agg_dir)
    logger.info("  已使用目录      : %s", args.used_dir)
    logger.info("  输出目录        : %s", args.unused_dir)
    logger.info("  原始数据        : %s", args.raw_dir)
    logger.info("  培养时长        : %s 小时", args.hours)
    logger.info("  可视化图        : %s", "关闭" if args.no_plots else "开启")
    logger.info("  已存在策略      : %s", args.on_exist)
    logger.info("  模式            : %s", "核对（不写入）" if args.dry_run else "写入")

    # ---------- 1. 索引全部数据组 ----------
    entries = load_group_index(args.agg_dir, logger=logger, batch_name=args.batch_dir)
    if not entries:
        logger.error("未扫到任何 selected__groups_*.xlsx，终止")
        return 1
    logger.info("-" * 74)

    # ---------- 2. 判定已使用 ----------
    used = collect_used(entries, args.used_dir, logger)
    logger.info("-" * 74)

    # ---------- 3. 扫描输出目录（幂等） ----------
    unused_root = Path(args.unused_dir)
    processed, orphans = scan_existing(unused_root, entries, logger)
    if processed:
        logger.info("已存在且完整的数据组文件夹：%d 个", len(processed))

    # ---------- 4. 逐个处理未使用的数据组 ----------
    todo = [e for e in sorted(entries, key=lambda x: (x.condition, x.index))
            if (e.condition, e.index) not in used]
    logger.info("数据组总数 %d｜已使用 %d｜待导出 %d",
                len(entries), len(used), len(todo))

    taken = {p.name for p in unused_root.iterdir()} if unused_root.is_dir() else set()
    pptx_cache = {}
    success, failed, skipped = [], [], []
    reused = set()          # 本次复用的残缺文件夹名

    logger.info("-" * 74)
    for n, entry in enumerate(todo, start=1):
        key = (entry.condition, entry.index)
        tag = f"[{n}/{len(todo)}] {entry.condition} 组{entry.index}"

        if key in processed:
            logger.info("%s 已存在，跳过：%s", tag, processed[key].name)
            skipped.append((entry, processed[key].name))
            continue

        mu = read_summary_metric(entry.path, config.UNUSED_SUMMARY_METRIC)
        if mu is None:
            mu = entry.mean_mu
            if mu is None:
                logger.warning("%s 读不到 %s，无法命名，跳过",
                               tag, config.UNUSED_SUMMARY_METRIC)
                failed.append((entry, None, ["无法读取 μmax 均值"]))
                continue
            logger.warning("%s summary 无 %s，改用组内 μmax 均值 %.6f",
                           tag, config.UNUSED_SUMMARY_METRIC, mu)
        base = mu_folder_name(mu)
        if base is None:
            failed.append((entry, None, ["μmax 无法转换为文件夹名"]))
            continue

        # 命名：优先复用上次中途失败留下的残缺文件夹，否则分配一个不冲突的名字
        if base in orphans:
            folder = orphans[base]
            reused.add(base)
        else:
            name, k = base, 2
            while name in taken:
                name = f"{base}_{k:02d}"
                k += 1
            folder = unused_root / name
        taken.add(folder.name)

        logger.info("%s → 文件夹「%s」（μmax=%.6f，%d 个腔室）",
                    tag, folder.name, mu, entry.size)
        try:
            res = build_one(entry, folder, args, logger, pptx_cache)
        except Exception as exc:                                # noqa: BLE001
            logger.exception("%s 处理失败：%s", tag, exc)
            failed.append((entry, folder.name, [f"异常：{exc}"]))
            continue

        if res["ok"]:
            success.append((entry, folder.name, res))
        else:
            failed.append((entry, folder.name, res["notes"] or ["产物不完整"]))

    rawdata.close_cache()

    # ------------------------------------------------------------------ 汇总日志
    logger.info("=" * 74)
    logger.info("成功清单（%d）", len(success))
    for entry, name, res in success:
        logger.info("  ■ %-8s ← %s 组%d（n=%d）| %s | %s | pptx 第 %s 页",
                    name, entry.condition, entry.index, entry.size,
                    res["xlsx"], res["copied"],
                    "/".join(str(p) for p in res["pptx"][1]))
    if failed:
        logger.info("-" * 74)
        logger.info("失败清单（%d）", len(failed))
        for entry, name, notes in failed:
            logger.info("  ✗ %s 组%d → %s：%s",
                        entry.condition, entry.index, name or "（未建文件夹）",
                        "；".join(notes))
    if skipped:
        logger.info("-" * 74)
        logger.info("已存在而跳过（%d）", len(skipped))
        for entry, name in skipped:
            logger.info("  · %s 组%d → %s", entry.condition, entry.index, name)
    if orphans:
        leftover = sorted(n for n in orphans if n not in reused)
        if leftover:
            logger.info("-" * 74)
            logger.info("残留的残缺文件夹（需人工确认）：%s", "、".join(sorted(leftover)))

    logger.info("-" * 74)
    logger.info("合计：成功 %d，失败 %d，跳过 %d（源目录未做任何改动）",
                len(success), len(failed), len(skipped))
    logger.info("输出目录：%s", unused_root)
    logger.info("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
