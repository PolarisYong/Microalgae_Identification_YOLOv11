# -*- coding: utf-8 -*-
"""
入口⑤：按指定条件改写一个数据组的全部标签
==========================================
把一个数据组文件夹（工作簿 + PPT + 文件名）里所有条件标识与原始腔室溯源，
统一改写为给定条件，用于「把该组数据视作某个条件的结果」的场景。

改写清单
--------
    工作簿内嵌图标题    L120-N300-IC5.25%_CB1 Cell growth curve fitting
                    → L182-N287-IC461_CB1 Cell growth curve fitting   （重绘 PNG）
    汇总页合并拟合图    (merged data, 12 chambers, L120-N300-IC5.25%)
                    → (merged data, 12 chambers, L182-N287-IC461)
    腔室页签名          CH1_CB1 … CH1_CB12        → CB1 … CB12
    汇总表腔室名称       CH1_CB1 … CH1_CB12        → CB1 … CB12
    腔室页 A 列图片名     CH11_CB1_H1.tif           → CB1_H1.tif
    「数据来源」页签      整页删除
    PPT 图内标题         同工作簿（重绘并换图，版式位置不动）
    PPT 腔室标签         CH11-CB1 …               → CB1 …
    文件名              2208_可视化结果.xlsx      → {条件}_可视化结果.xlsx
                        L120‑N300‑IC5.25%.pptx   → {条件}.pptx

数据本身（目标数量 / 总面积 / 拟合参数 / 合并拟合结果）逐格保持不变；
不改写文件夹名；selected__groups_*.xlsx 原样保留。
改写前整份备份到 <项目根>/_backup/。

用法
----
    python relabel_group.py                              # 用下面的默认值
    python relabel_group.py --condition "L182‑N287‑IC461"
    python relabel_group.py --src "…/2208" --dry-run     # 只打印计划，不写文件
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                             # noqa: E402
from core import logging_setup, plotting, relabel, writer  # noqa: E402
from core.fitting import fit_curve                         # noqa: E402

# 默认目标：20组模拟数据+1组最优模拟 里的那一组
DEFAULT_SRC = config.PART1_ROOT / "20组模拟数据+1组最优模拟" / "2208"
# 默认新条件（连字符用 U+2011，与既有工况目录名一致；图标题里会自动换成 ASCII '-'）
DEFAULT_CONDITION = "L182\u2011N287\u2011IC461"
# 备份根目录（放在项目根，不属于任何数据目录）
BACKUP_ROOT = config.PROJECT_ROOT / "_backup"
# PPT 配图单独用高分辨率重绘（原 PPT 里的图就是 300 dpi）
PPT_DPI = 300


# --------------------------------------------------------------------- 文件定位
def find_assets(src_dir, logger=None):
    """在数据组文件夹里定位工作簿 / PPT / 分组索引文件。"""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"数据组文件夹不存在：{src_dir}")

    xlsxs = [p for p in sorted(src_dir.glob("*.xlsx"))
             if not p.name.startswith("~$") and not p.name.startswith("selected__")]
    pptxs = [p for p in sorted(src_dir.glob("*.pptx")) if not p.name.startswith("~$")]
    selected = [p for p in sorted(src_dir.glob("selected__groups_*.xlsx"))]

    if len(xlsxs) != 1:
        raise ValueError(f"{src_dir.name} 下应有且仅有 1 个数据工作簿，"
                         f"实际找到 {len(xlsxs)} 个：{[p.name for p in xlsxs]}")
    if len(pptxs) != 1:
        raise ValueError(f"{src_dir.name} 下应有且仅有 1 个 PPT，"
                         f"实际找到 {len(pptxs)} 个：{[p.name for p in pptxs]}")
    if logger:
        logger.info("定位到工作簿 %s｜PPT %s｜分组索引 %s",
                    xlsxs[0].name, pptxs[0].name,
                    ", ".join(p.name for p in selected) or "（无）")
    return xlsxs[0], pptxs[0], selected


# --------------------------------------------------------------------- 备份
def backup(src_dir):
    """整份复制源文件夹到 _backup/<名字>_原版_<时间戳>/，返回备份路径。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_ROOT / f"{src_dir.name}_原版_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst)
    return dst


# --------------------------------------------------------------------- PPT 配图
def render_ppt_pngs(chambers, condition, logger=None, tag=""):
    """按新条件重绘每个腔室的「生长曲线拟合」图（PPT 用，高分辨率）。"""
    import numpy as np

    old_dpi = config.PLOT_DPI
    config.PLOT_DPI = PPT_DPI
    try:
        pngs = []
        for ch in chambers:
            counts = np.asarray([v for v in ch["counts"] if v is not None], dtype=float)
            t = np.arange(counts.size, dtype=float)
            fit = fit_curve(t, counts)
            buf = plotting.chamber_fit_figure(t, counts, fit, ch["plot_name"])
            pngs.append(buf.getvalue())
        if logger:
            logger.info("%s 重绘 PPT 配图 %d 张（%d dpi）", tag, len(pngs), PPT_DPI)
        return pngs
    finally:
        config.PLOT_DPI = old_dpi


# --------------------------------------------------------------------- 校验
def verify(new_xlsx, original_summary, chamber_count, logger, tag=""):
    """改写后逐格比对「汇总」页 + 检查页签名 / 图数 / 残留溯源。"""
    import openpyxl

    problems = []
    # 普通模式加载：read_only 模式下读不到 _images（内嵌图清单）
    wb = openpyxl.load_workbook(new_xlsx, data_only=True)
    try:
        names = wb.sheetnames
        expected = ["汇总"] + [f"CB{i}" for i in range(1, chamber_count + 1)]
        if names != expected:
            problems.append(f"页签名不符：{names} ≠ {expected}")
        if relabel.SOURCE_SHEET in names:
            problems.append("「数据来源」页签未被删除")

        new_summary = [[v for v in r] for r in wb[relabel.SUMMARY_SHEET].iter_rows(values_only=True)]
        if len(new_summary) != len(original_summary):
            problems.append(f"汇总页行数 {len(new_summary)} ≠ 原 {len(original_summary)}")
        else:
            for ri, (old_row, new_row) in enumerate(zip(original_summary, new_summary), 1):
                for ci, (a, b) in enumerate(zip(old_row, new_row), 1):
                    if a == b:
                        continue
                    # 只允许「腔室名称」列随页签改名而变
                    if ci == 1 and isinstance(a, str) and relabel.CHAMBER_KEY_RE.fullmatch(a):
                        continue
                    problems.append(f"汇总页 R{ri}C{ci}：{a!r} → {b!r}")
        images = len(wb[relabel.SUMMARY_SHEET]._images)
        for i in range(1, chamber_count + 1):
            images += len(wb[f"CB{i}"]._images)
        expected_images = chamber_count * 4 + 3
        if images != expected_images:
            problems.append(f"图数 {images} ≠ 预期 {expected_images}")
    finally:
        wb.close()

    texts = relabel.collect_cell_texts(
        new_xlsx, skip_sheets=[relabel.SOURCE_SHEET])
    hits = relabel.find_leftovers(texts, [relabel.SOURCE_PREFIX_RE,
                                          relabel.CHAMBER_KEY_RE,
                                          relabel.PPT_LABEL_RE])
    if hits:
        problems.append(f"单元格仍残留原始溯源 {len(hits)} 处：{hits[:5]}")

    if logger:
        if problems:
            for item in problems:
                logger.warning("%s 校验不通过：%s", tag, item)
        else:
            logger.info("%s 校验通过：页签 %s｜图 %d 张｜数据与原版逐格一致",
                        tag, "汇总+CB1..CB%d" % chamber_count, images)
    return problems


# --------------------------------------------------------------------- 主流程
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="按指定条件改写一个数据组的全部标签（工作簿 / PPT / 文件名）")
    parser.add_argument("--src", default=str(DEFAULT_SRC),
                        help="数据组文件夹（默认 20组模拟数据+1组最优模拟/2208）")
    parser.add_argument("--condition", default=DEFAULT_CONDITION,
                        help="新条件标识，如 L182‑N287‑IC461")
    parser.add_argument("--note", default="",
                        help="条件的可读描述，仅写入日志，如 '光照=182 μmol/m²/s'")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印改写计划与校验结果，不写任何文件")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份（不推荐）")
    parser.add_argument("--keep-old", action="store_true",
                        help="保留原文件名（默认删除被改名的旧文件）")
    args = parser.parse_args(argv)

    logger = logging_setup.setup_logger(Path(__file__).resolve().parent / "_logs",
                                        name="relabel_group")
    src_dir = Path(args.src)
    condition = args.condition
    tag = f"[{src_dir.name}]"

    logger.info("=" * 68)
    logger.info("改写目标：%s → 条件 %s%s", src_dir, condition,
                f"（{args.note}）" if args.note else "")
    logger.info("=" * 68)

    try:
        xlsx, pptx, selected = find_assets(src_dir, logger)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("定位失败：%s", exc)
        return 2

    new_xlsx = src_dir / f"{condition}{config.GENERATED_SUFFIX}"
    new_pptx = src_dir / config.PPTX_NAME_TEMPLATE.format(condition=condition)

    logger.info("计划：工作簿 %s → %s", xlsx.name, new_xlsx.name)
    logger.info("计划：PPT   %s → %s", pptx.name, new_pptx.name)
    logger.info("计划：分组索引 %s 保持原样",
                ", ".join(p.name for p in selected) or "（无）")

    # ---- 1. 读回腔室数据（不触碰原始 tif，数据一字不改） ----
    chambers, original_summary = relabel.read_source(xlsx, condition, logger, tag)
    logger.info("%s 读回 %d 个腔室页签，共 %d 个数据点",
                tag, len(chambers), sum(len(c["counts"]) for c in chambers))

    if args.dry_run:
        logger.info("%s dry-run：不写文件。页签规划 %s",
                    tag, "汇总 + " + ", ".join(c["new_name"] for c in chambers))
        logger.info("%s dry-run：图标题规划 %s",
                    tag, " / ".join(c["plot_name"] for c in chambers[:2]) + " …")
        return 0

    # ---- 2. 备份 ----
    if args.no_backup:
        logger.warning("%s 已跳过备份（--no-backup）", tag)
    else:
        dst = backup(src_dir)
        logger.info("%s 已备份原版 → %s", tag, dst)

    # ---- 3. 重建工作簿 ----
    writer.build_workbook(new_xlsx, chambers, logger=logger, tag=tag,
                          condition=condition, include_source=False,
                          plots=config.PLOT_ENABLED)

    # ---- 4. 重建 PPT ----
    pngs = render_ppt_pngs(chambers, condition, logger, tag)
    pic_count, label_count = relabel.rebuild_pptx(pptx, new_pptx, pngs, logger, tag)
    logger.info("%s PPT 换图 %d 张、改写标签 %d 个 → %s",
                tag, pic_count, label_count, new_pptx.name)

    # ---- 5. 删掉被改名的旧文件 ----
    if not args.keep_old:
        for old in (xlsx, pptx):
            old.unlink()
            logger.info("%s 已删除旧文件 %s", tag, old.name)

    # ---- 6. 校验 ----
    problems = verify(new_xlsx, original_summary, len(chambers), logger, tag)
    if problems:
        logger.error("%s 改写完成但有 %d 项校验未通过，请查看日志", tag, len(problems))
        return 1

    logger.info("-" * 68)
    logger.info("完成：工作簿 %s｜PPT %s｜腔室 %d 个｜图 %d 张",
                new_xlsx.name, new_pptx.name, len(chambers), len(chambers) * 4 + 3)
    logger.info("统计：读取腔室 %d 个｜重绘图 %d 张（工作簿 %d + PPT %d）｜删除旧文件 %d 个",
                len(chambers), len(chambers) * 4 + 3 + len(pngs),
                len(chambers) * 4 + 3, len(pngs), 0 if args.keep_old else 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
