# -*- coding: utf-8 -*-
"""
二次分组的表格写出
==================
负责三份文档：

    未参与数据分组腔室数据.xlsx   —— 全部未标红腔室的原始页签（每份 ≤ 50 个页签，超出拆分）
    剔除腔室数据.xlsx             —— 被生长规律筛查剔除的腔室（同上，附剔除原因）
    文档数据来源.xlsx             —— 逐条记录每组成员的实际来源 + 剔除清单 + 全量腔室清单

每个腔室页签在 H/I 两列标注腔室实际来源（自变量文件夹 / 源文件 / 原始页签），
不改动 A:F 的原始数据结构，保证页签仍可被后续程序按原格式读取。
"""
from __future__ import annotations

import math
from pathlib import Path

import config

# 来源标注写入的列（H/I），与原始数据的 A:F 互不干扰
ANNOTATE_COL = 8
VALUE_COL = 9


def mu_folder_name(value, digits=None):
    """μmax 均值 → 文件夹名：去掉前导「0.」后取前 digits 位（0.0657… → 0657）。"""
    digits = digits or config.UNUSED_FOLDER_DIGITS
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    text = f"{abs(v):.10f}"
    int_part, _, frac = text.partition(".")
    digits_part = (int_part + frac) if int_part not in ("0", "") else frac
    return digits_part[:digits] or "0000"


def make_sheet_name(ref, used, short_file=None, short_cond=None):
    """
    为腔室页签取一个 ≤31 字符且在工作簿内唯一的名字。

    依次尝试：原始页签名 → 「源文件短名_页签名」→「工况_源文件短名_页签名」，
    仍冲突则追加 _02 / _03 …
    """
    from core.group_assets import short_condition, short_file as _short_file

    short_file = (short_file if short_file is not None
                  else _short_file(ref.src_file))
    short_cond = (short_cond if short_cond is not None
                  else short_condition(ref.condition))

    candidates = [
        ref.sheet,
        f"{short_file}_{ref.sheet}",
        f"{short_cond}_{short_file}_{ref.sheet}",
    ]
    for cand in candidates:
        name = _sanitize(cand[:31])
        if name and name not in used:
            used.add(name)
            return name
    base = _sanitize(candidates[-1][:28])
    k = 2
    while True:
        name = f"{base}_{k:02d}"
        if name not in used:
            used.add(name)
            return name
        k += 1


def _sanitize(name: str) -> str:
    """去掉 Excel 页签名不允许的字符，并去掉首尾空白。"""
    for ch in '[]:*?/\\':
        name = name.replace(ch, "-")
    return name.strip()


def _annotate(ws, ref, reason=None):
    """在 H/I 列写入来源标注（不改动原始数据的 A:F 列）。"""
    from openpyxl.styles import Font

    pairs = [
        ("自变量文件夹", ref.condition),
        ("源文件", ref.src_file),
        ("原始页签", ref.sheet),
    ]
    if reason:
        pairs.append(("剔除原因", reason))
    for i, (label, value) in enumerate(pairs, start=1):
        c = ws.cell(row=i, column=ANNOTATE_COL, value=label)
        c.font = Font(bold=True)
        ws.cell(row=i, column=VALUE_COL, value=value)
    ws.column_dimensions["H"].width = 14
    ws.column_dimensions["I"].width = 44


def write_chamber_workbooks(refs, out_dir, stem, logger=None, tag="", reasons=None):
    """
    把腔室页签写成工作簿；超过 MAX_SHEETS_PER_FILE 个页签自动拆分。

    reasons : {(condition, src_file, sheet): 剔除原因}（可选）

    返回写出的文件路径列表。
    """
    import openpyxl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = list(refs)
    if not refs:
        if logger:
            logger.info("%s 无腔室可写入：%s", tag, stem)
        return []

    per = max(1, int(config.MAX_SHEETS_PER_FILE))
    n_parts = math.ceil(len(refs) / per)
    written = []
    used_names = set()

    for p in range(n_parts):
        chunk = refs[p * per:(p + 1) * per]
        if n_parts == 1:
            name = stem
        elif p == 0 and config.SPLIT_FIRST_PART_PLAIN:
            name = stem
        else:
            name = f"{stem}_part{p + 1}"
        path = out_dir / f"{name}.xlsx"

        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for ref in chunk:
            sheet_name = make_sheet_name(ref, used_names)
            ws = wb.create_sheet(sheet_name)
            for r in ref.rows or []:
                ws.append(list(r))
            _annotate(ws, ref, (reasons or {}).get(ref.uid))
            ws.freeze_panes = "A2"
        wb.save(path)
        wb.close()
        written.append(path)
        if logger:
            logger.info("%s 写出 %s（%d 个腔室页签）", tag, path.name, len(chunk))

    if logger and n_parts > 1:
        logger.info("%s %s 共 %d 个页签，按每份 %d 个拆成 %d 个文件",
                    tag, stem, len(refs), per, n_parts)
    return written


def to_group_rows(refs):
    """ChamberRef → 供 batch_grouping 分组算法使用的字典列表。"""
    rows = []
    for ref in refs:
        rows.append({
            "source_file": f"{ref.condition}/{ref.src_file}",
            "summary_row": ref.summary_row,
            "chamber": ref.sheet,
            "flow": f"CH{ref.flow_no}",
            "flow_no": ref.flow_no,
            "chamber_no": ref.chamber_no,
            "umax": ref.umax,
            "krel": ref.krel,
            "n0": ref.n0,
            "r2": ref.r2,
            "td": ref.td,
            "f": ref.f,
            "eta": ref.eta,
            "lag": ref.lag,
            "log": ref.log,
            "stable": ref.stable,
        })
    return rows


def write_doc_source(path, groups, excluded, all_refs, strict_keys, logger=None):
    """
    写出「文档数据来源.xlsx」。

    groups    : [(folder_name, group_index, [ChamberResult], stats_dict)]
    excluded  : [(ChamberRef, 原因)]
    all_refs  : 全部未参与分组的腔室
    strict_keys: 进入 batch_grouping 严格候选的 uid 集合
    """
    import openpyxl
    from openpyxl.styles import Alignment, Font

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def _new(title, headers, widths):
        ws = wb.create_sheet(title)
        ws.append(headers)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal="center", vertical="center")
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
        return ws

    # ---------- ① 数据组成员 ----------
    ws = _new("数据组成员",
              ["数据组文件夹", "组序号", "组内编号", "页签名（新）", "μmax (h^-1)", "R²",
               "自变量文件夹", "源文件", "原始页签"],
              [14, 8, 10, 14, 14, 10, 20, 34, 12])
    for folder_name, g_index, rows, refs_by_key in groups:
        for i, row in enumerate(sorted(rows, key=lambda r: (r.flow_no, r.chamber_no,
                                                            r.source_file)), start=1):
            ref = refs_by_key[(row.source_file, row.chamber)]
            ws.append([folder_name, g_index, i, f"CH1_CB{i}", row.umax, row.r2,
                       ref.condition, ref.src_file, ref.sheet])

    # ---------- ② 未参与分组腔室 ----------
    ws2 = _new("未参与分组腔室",
               ["自变量文件夹", "源文件", "原始页签", "μmax (h^-1)", "R²", "Nrel(0)",
                "是否参与分组", "说明"],
               [20, 34, 12, 14, 10, 12, 14, 34])
    excluded_keys = {ref.uid: reason for ref, reason in excluded}
    for ref in sorted(all_refs, key=lambda r: (r.condition, r.src_file, r.sheet)):
        if ref.uid in excluded_keys:
            joined, note = "否", "生长规律筛查剔除"
        elif ref.uid in strict_keys:
            joined, note = "是", "进入 batch_grouping 严格候选"
        else:
            joined, note = "否", "未达 batch_grouping 严格候选条件（N0>0 且 R²>0.9）"
        ws2.append([ref.condition, ref.src_file, ref.sheet, ref.umax, ref.r2, ref.n0,
                    joined, note])

    # ---------- ③ 剔除腔室 ----------
    ws3 = _new("剔除腔室",
               ["自变量文件夹", "源文件", "原始页签", "μmax (h^-1)", "R²", "剔除原因"],
               [20, 34, 12, 14, 10, 60])
    for ref, reason in sorted(excluded, key=lambda x: (x[0].condition, x[0].src_file,
                                                       x[0].sheet)):
        ws3.append([ref.condition, ref.src_file, ref.sheet, ref.umax, ref.r2, reason])

    wb.save(path)
    wb.close()
    if logger:
        logger.info("文档数据来源：%s（成员 %d 条 / 腔室 %d 条 / 剔除 %d 条）",
                    path.name,
                    sum(len(g[2]) for g in groups), len(all_refs), len(excluded))
    return path
