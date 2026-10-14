# -*- coding: utf-8 -*-
"""
定位并读取原始腔室页签
======================
batch_grouping 记录的是聚合后的文件名（如 `20260712_CH7_可视化结果_96.xlsx`），
本模块把它还原成原始数据目录里的真实路径：

    μmax_revision_原始数据/<YYYYMMDD>/数据汇总/03_可视化结果/<N>小时/CH<k>_可视化结果_<N>.xlsx

再从该工作簿取出 `CH<k>_CB<j>` 页签的全部单元格值。
"""
import re
from pathlib import Path

import config

_WB_CACHE = {}   # 路径 -> workbook，避免同一文件反复打开


def parse_source_file(fname: str):
    """
    '20260712_CH7_可视化结果_96.xlsx' → ('20260712', 'CH7', 96)
    解析失败返回 None。
    """
    m = re.match(r"^(\d{8})_(CH\d+)_可视化结果_(\d+)\.xlsx$", fname.strip())
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def resolve_raw_path(fname, raw_dir=None, hours=None):
    """聚合文件名 → 原始数据文件的绝对路径；找不到返回 None。"""
    from pathlib import Path
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    hours = hours if hours is not None else config.CULTURE_HOURS

    parsed = parse_source_file(fname)
    if parsed is None:
        return None
    date, ch, _ = parsed
    p = (raw_dir / date / "数据汇总" / "03_可视化结果"
         / f"{hours}小时" / f"{ch}_可视化结果_{hours}.xlsx")
    return p if p.exists() else None


def _open(path):
    import openpyxl
    key = str(path)
    if key not in _WB_CACHE:
        _WB_CACHE[key] = openpyxl.load_workbook(path, data_only=True)
    return _WB_CACHE[key]


def read_chamber_sheet(path, sheet_name, logger=None, tag=""):
    """
    读取某个腔室页签。
    返回 (rows, params)
        rows  : 全部单元格值的二维列表（原样搬运，含表头）
        params: {参数名: 参数值}（来自「参数名 / 参数值」两列）
    """
    wb = _open(path)
    if sheet_name not in wb.sheetnames:
        if logger:
            logger.warning("%s 文件 %s 中缺少页签 %s，已跳过",
                           tag, path.name, sheet_name)
        return None, None

    ws = wb[sheet_name]
    rows = [[c for c in r] for r in ws.iter_rows(values_only=True)]

    params = {}
    for r in rows:
        if len(r) >= 6 and r[4] is not None:
            key = str(r[4]).strip()
            if key and key != "参数名":
                params[key] = r[5]
    return rows, params


def extract_column(rows, column_index, max_points=None):
    """
    从腔室页签的二维单元格值中取出某一列。

    表头行（第 1 行）跳过；空单元格跳过。`max_points` 用于按培养时长截断
    （例如 96 小时 + 0 小时共 97 点），传 None 表示不截断。
    """
    values = []
    for r in rows[1:]:
        if len(r) <= column_index:
            continue
        v = r[column_index]
        if v is None:
            continue
        values.append(v)
    if max_points is not None:
        values = values[:max_points]
    return values


def extract_counts(rows, max_points=None):
    """「目标数量」列（相对数量 Nrel），供拟合使用。"""
    return extract_column(rows, config.COUNTS_COLUMN_INDEX, max_points)


def extract_areas(rows, max_points=None):
    """
    「总面积(μm²)」列。该列在原始数据中已按 0 h 归一化，
    可直接作为「相对总细胞面积」使用（与参考实现口径一致）。
    """
    return extract_column(rows, config.AREA_COLUMN_INDEX, max_points)


def relative_average_area(areas, counts):
    """
    相对平均细胞面积 = 总面积 / 目标数量（逐点）。
    对应原始工作簿「数据汇总」页签里的「相对平均细胞面积」列，
    等价于 numpy 的 areas / counts，除以 0 的点返回 NaN。
    """
    import numpy as np

    a = np.asarray(areas, dtype=float)
    c = np.asarray(counts, dtype=float)
    n = min(a.size, c.size)
    a, c = a[:n], c[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(c != 0, a / c, np.nan)
    return out


def close_cache():
    for wb in _WB_CACHE.values():
        try:
            wb.close()
        except Exception:
            pass
    _WB_CACHE.clear()
