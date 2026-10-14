# -*- coding: utf-8 -*-
"""
把「汇总结果」页的腔室汇总表 + 培养条件块搬进 selected_N 页
==========================================================
selected__groups_X_Y.xlsx 的 selected_Y 页签原本只有一张 13 列腔室级参数表
（Source file / Flow channel / Chamber / μmax / K / N0 / R² / Td / F / η /
Tlag / Tprolif / Tplat），与英文版可视化工作簿「汇总结果」页的表结构对不上。

本模块把「汇总结果」页 A1:H<培养条件取值行> 整块（8 列腔室级汇总表 +
空行 + Cultivation conditions (coded) 标题 + 三列条件表头 + 三列条件取值）
复制到 selected_N 页替换原表，并把该页嵌的组合生长曲线图整体下移，
同时删除 selected_N 以外的页签。

复制口径：
- 逐格照搬 value / font / alignment / border / fill / number_format /
  protection；RichText（斜体变量、上下标）深拷贝为独立 InlineFont，
  避免与源对象共享
- 复制前把 selected_N 页已用区域全部清空并复位样式，避免残留旧表
- 列宽 A:H 取「汇总结果」页；行 1 高度 44，其余行高复位后按源页设置
- 图片锚点 OneCellAnchor._from.row 下移到条件块末行之下第 2 行
  （末行 R → 锚点 Excel 行 R+2），原图所在 "高行"（如 212）随图搬到新行，
  旧行高复位，保证图上移下后不压住表格
- 除 selected_N 外的页签直接删除

约定：源「汇总结果」页只读，不改动。
"""
from copy import copy
from pathlib import Path

SUMMARY_SHEET_EN = "汇总结果"
COND_PREFIX = "Cultivation conditions"
REGION_COLS = 8          # 汇总结果页表宽 A:H
ANCHOR_GAP = 1           # 条件块末行与图片锚点之间留 1 个空行


# --------------------------------------------------------------------- 定位
def find_region(path_or_ws, logger=None, tag=""):
    """
    返回 (ws, bottom_row)；bottom_row = 培养条件取值行（即要复制的末行）。
    源页签必须存在 "Cultivation conditions (...)" 标题行。
    """
    import openpyxl

    if isinstance(path_or_ws, (str, Path)):
        wb = openpyxl.load_workbook(path_or_ws, rich_text=True)
        ws = wb[SUMMARY_SHEET_EN]
    elif isinstance(path_or_ws, openpyxl.workbook.workbook.Workbook):
        wb = path_or_ws
        ws = wb[SUMMARY_SHEET_EN]
    else:
        ws = path_or_ws

    title_row = None
    for r in range(1, ws.max_row + 1):
        v = ws.cell(row=r, column=1).value
        text = v if isinstance(v, str) else (
            str(v) if v is not None and not hasattr(v, "_r") else "")
        if text.strip().startswith(COND_PREFIX):
            title_row = r
            break
    if title_row is None:
        raise ValueError(f"{SUMMARY_SHEET_EN} 页找不到 {COND_PREFIX!r} 标题行")

    bottom = title_row + 2
    vals = [ws.cell(row=bottom, column=c).value for c in range(1, 4)]
    if any(v is None or str(v).strip() == "" for v in vals):
        raise ValueError(f"第 {bottom} 行不像培养条件取值行：{vals}")
    if logger:
        logger.info("  %s %s 页复制区域 A1:H%d（条件块第 %d 行）",
                    tag, SUMMARY_SHEET_EN, bottom, title_row)
    return ws, bottom


# --------------------------------------------------------------------- 富文本
def _clone_rich(value):
    """CellRichText 深拷贝（TextBlock + InlineFont 均为新对象）。"""
    from openpyxl.cell.rich_text import CellRichText, TextBlock

    blocks = []
    for block in value:
        if isinstance(block, TextBlock):
            blocks.append(TextBlock(copy(block.font), block.text))
        elif isinstance(block, str):
            blocks.append(block)
        else:
            blocks.append(copy(block))
    return CellRichText(blocks)


# --------------------------------------------------------------------- 清空 / 复制
def _reset_sheet(ws):
    """清空已用区域的值并把样式复位为默认。"""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection

    empty = dict(font=Font(), border=Border(), fill=PatternFill(),
                 alignment=Alignment(), protection=Protection())
    last_row = max(ws.max_row, 1)
    last_col = max(ws.max_column, 1)
    for row in ws.iter_rows(min_row=1, max_row=last_row, max_col=last_col):
        for cell in row:
            cell.value = None
            for attr, obj in empty.items():
                setattr(cell, attr, obj)
            cell.number_format = "General"
    for idx in list(ws.row_dimensions):
        ws.row_dimensions[idx].height = None
    return last_row, last_col


def _copy_region(src_ws, dst_ws, n_rows, n_cols=REGION_COLS):
    """把 src_ws 的 A1:<n_cols><n_rows> 逐格复制到 dst_ws 同位置。"""
    from openpyxl.cell.rich_text import CellRichText

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            src = src_ws.cell(row=r, column=c)
            dst = dst_ws.cell(row=r, column=c)
            value = src.value
            if isinstance(value, CellRichText):
                value = _clone_rich(value)
            dst.value = value
            dst.font = copy(src.font)
            dst.alignment = copy(src.alignment)
            dst.border = copy(src.border)
            dst.fill = copy(src.fill)
            dst.number_format = src.number_format
            dst.protection = copy(src.protection)


def _copy_layout(src_ws, dst_ws, n_rows, n_cols=REGION_COLS):
    """列宽 A:<n_cols> 与行高 1:<n_rows> 照搬。"""
    from openpyxl.utils import get_column_letter

    for c in range(1, n_cols + 1):
        letter = get_column_letter(c)
        src = src_ws.column_dimensions.get(letter)
        if src is not None and src.width:
            dst_ws.column_dimensions[letter].width = src.width
    for r in range(1, n_rows + 1):
        src = src_ws.row_dimensions.get(r)
        if src is not None and src.height:
            dst_ws.row_dimensions[r].height = src.height


# --------------------------------------------------------------------- 图片下移
def _move_images(ws, new_row_0based, band_height=None, logger=None, tag=""):
    """
    把页内图片的 OneCellAnchor 起点行改成 new_row_0based（0 基）。
    返回 (移动张数, 旧锚点行列表)。
    """
    moved, old_rows = 0, []
    for img in list(getattr(ws, "_images", [])):
        anchor = getattr(img, "anchor", None)
        marker = getattr(anchor, "_from", None)
        if marker is None:                     # TwoCellAnchor 等其它形式
            continue
        old_rows.append(marker.row)
        marker.row = new_row_0based
        marker.col = 0
        marker.colOff = 0
        marker.rowOff = 0
        moved += 1
    if band_height and moved:
        ws.row_dimensions[new_row_0based + 1].height = band_height
    if logger and moved:
        logger.info("  %s 组合图下移：锚点行 %s → Excel 第 %d 行（行高 %s）",
                    tag, old_rows, new_row_0based + 1, band_height)
    return moved, old_rows


# --------------------------------------------------------------------- 主流程
def apply_cond_block(selected_path, viz_path, logger=None, tag="",
                     dry_run=False):
    """
    把 viz_path「汇总结果」页的条件块区并入 selected_path 的 selected_N 页。
    返回统计 dict：{sheet, region_bottom, old_sheets, images, rows_cleared}
    """
    import openpyxl

    selected_path = Path(selected_path)
    viz_path = Path(viz_path)

    # ---------- 读源区 ----------
    src_wb = openpyxl.load_workbook(viz_path, rich_text=True)
    try:
        src_ws, bottom = find_region(src_wb, logger=logger, tag=tag)
    except BaseException:
        src_wb.close()
        raise

    stats = {"sheet": None, "region_bottom": bottom, "old_sheets": [],
             "images": 0, "cleared_rows": 0, "cleared_cols": 0}

    wb = openpyxl.load_workbook(selected_path, rich_text=True)
    try:
        sheet = next((n for n in wb.sheetnames if n.startswith("selected_")),
                     None)
        if sheet is None:
            raise ValueError(f"{selected_path.name} 里没有 selected_ 页签")
        stats["sheet"] = sheet
        stats["old_sheets"] = [n for n in wb.sheetnames if n != sheet]

        if dry_run:
            return stats

        ws = wb[sheet]
        band_height = ws.row_dimensions[20].height or 212.0
        rows_cleared, cols_cleared = _reset_sheet(ws)
        stats["cleared_rows"], stats["cleared_cols"] = rows_cleared, cols_cleared

        _copy_region(src_ws, ws, bottom)
        _copy_layout(src_ws, ws, bottom)

        # 图片锚点：条件块末行 R → 空 1 行 → 锚点 Excel 行 R + 2
        anchor_row0 = bottom + ANCHOR_GAP
        moved, _ = _move_images(ws, anchor_row0, band_height,
                                logger=logger, tag=tag)
        stats["images"] = moved

        # 删除其余页签
        for name in stats["old_sheets"]:
            del wb[name]
        wb.active = wb.sheetnames.index(sheet)
        wb.save(selected_path)
    finally:
        wb.close()
        src_wb.close()
    if logger:
        logger.info("  %s %s 页已替换为汇总结果 A1:H%d；删除页签 %s；下移图 %d 张",
                    tag, stats["sheet"], bottom,
                    "/".join(stats["old_sheets"]) or "无", stats["images"])
    return stats
