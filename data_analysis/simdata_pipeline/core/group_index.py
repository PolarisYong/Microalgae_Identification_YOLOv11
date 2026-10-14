# -*- coding: utf-8 -*-
"""
数据组索引与匹配
================
两侧数据的「键」不一样，必须先归一化才能比：

    selected__groups_<i>_<n>.xlsx（batch_grouping 产物，源侧）
        第 1 个页签 `selected_<n>`：
            源文件名称(source_file) = "20260531_CH1_可视化结果_96.xlsx"   ← 日期_文件名
            腔室名称(chamber)       = "CH1_CB5"

    20组模拟数据/<工况>/<工况>_可视化结果.xlsx（本管线产物，目标侧）
        「数据来源」页签：
            腔室名         = "CH1_CB1"          ← 重命名后的序号，不能用作键
            原始数据来源    = "20260531\\数据汇总\\03_可视化结果\\96小时\\CH4_可视化结果_96.xlsx｜页签：CH4_CB1"

归一化后两侧都是 (日期_文件名, 原腔室名)，可直接做**集合相等**比对。
按集合（而非逐行顺序）比对，是因为输出页签顺序是 CH1_CB1…CBn 的重排，
与分组文件里的顺序无关。
"""
import re
from pathlib import Path

# 分组文件名：selected__groups_<组序号>_<腔室数>.xlsx
GROUP_RE = re.compile(r"selected__groups_(\d+)_(\d+)\.xlsx", re.IGNORECASE)
# 「原始数据来源」里的文件名：形如 20260531\数据汇总\...\CH4_可视化结果_96.xlsx｜页签：CH4_CB1
SOURCE_SPLIT = "｜页签："

COL_SOURCE_FILE = "源文件名称(source_file)"
COL_CHAMBER = "腔室名称(chamber)"
COL_MUMAX = "最大比生长速率μmax (h^-1)"

# 输出工作簿「汇总」页签里用于回退匹配的两行标签
LABEL_N = "参与拟合的页签数量"
LABEL_MU = "最大比生长速率μmax (h^-1)"


# --------------------------------------------------------------------------- 键
def normalize_source_file(rel_path):
    """相对路径 → 分组文件里的 source_file 形式：`日期_文件名`。"""
    p = Path(str(rel_path).strip())
    parts = p.parts
    if len(parts) >= 2:
        return f"{parts[0]}_{p.name}"        # 20260531 + CH4_可视化结果_96.xlsx
    return p.name


def parse_source_record(text):
    """'路径｜页签：CH4_CB1' → ('日期_文件名', 'CH4_CB1')。"""
    text = str(text)
    if SOURCE_SPLIT not in text:
        return None
    path, sheet = text.split(SOURCE_SPLIT, 1)
    return normalize_source_file(path), sheet.strip()


def chambers_of_output(xlsx_path):
    """
    读输出工作簿的「数据来源」页签 → {(日期_文件名, 原腔室名)}。
    没有该页签返回 None（调用方改用回退匹配）。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        if "数据来源" not in wb.sheetnames:
            return None
        ws = wb["数据来源"]
        keys = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) < 2 or not row[1]:
                continue
            parsed = parse_source_record(row[1])
            if parsed:
                keys.add(parsed)
        return keys or None
    finally:
        wb.close()


def read_summary_metric(xlsx_path, metric, sheet=None):
    """
    读分组文件 summary 页签的某项指标：在 A 列找 metric，返回 B 列数值。
    summary 页签形如：
        A                        B
        metric                   value
        mu_max_mean_h^-1         0.0657777777777778
    找不到返回 None。
    """
    import openpyxl
    import config

    sheet = sheet or config.UNUSED_SUMMARY_SHEET
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        if sheet not in wb.sheetnames:
            return None
        ws = wb[sheet]
        for row in ws.iter_rows(values_only=True):
            if row and row[0] is not None and str(row[0]).strip() == metric:
                if len(row) > 1 and isinstance(row[1], (int, float)):
                    return float(row[1])
                return None
        return None
    finally:
        wb.close()


def _header_index(header_row, *candidates):
    """按列名找列下标；找不到返回 None。"""
    for name in candidates:
        for i, v in enumerate(header_row):
            if v is not None and str(v).strip() == name:
                return i
    return None


def group_signature(xlsx_path):
    """
    读分组文件 → (腔室键集合, 组均值 μmax, 腔室数)。
    腔室数取自文件名里的 <n>，并与其页签名 `selected_<n>` 交叉核对。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        sheet_name = wb.sheetnames[0]
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return set(), None, 0
        i_f = _header_index(rows[0], COL_SOURCE_FILE)
        i_c = _header_index(rows[0], COL_CHAMBER)
        i_mu = _header_index(rows[0], COL_MUMAX)
        if i_f is None or i_c is None:
            return set(), None, 0
        keys, mus = set(), []
        for r in rows[1:]:
            if len(r) <= max(i_f, i_c) or not r[i_f] or not r[i_c]:
                continue
            keys.add((str(r[i_f]).strip(), str(r[i_c]).strip()))
            if i_mu is not None and len(r) > i_mu and isinstance(r[i_mu], (int, float)):
                mus.append(float(r[i_mu]))
        mean_mu = sum(mus) / len(mus) if mus else None
        return keys, mean_mu, len(keys)
    finally:
        wb.close()


def output_signature_fallback(xlsx_path):
    """
    回退用的签名：（腔室数, 合并拟合 μmax）。
    仅在输出工作簿缺少「数据来源」页签时使用。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        if "汇总" not in wb.sheetnames:
            return None
        ws = wb["汇总"]
        values = {}
        for r in ws.iter_rows(values_only=True):
            if r and r[0] is not None:
                values.setdefault(str(r[0]).strip(), r[1] if len(r) > 1 else None)
        n = values.get(LABEL_N)
        mu = values.get(LABEL_MU)
        if n is None or mu is None:
            return None
        return int(n), float(mu)
    finally:
        wb.close()


# --------------------------------------------------------------------------- 索引
class GroupEntry:
    """一个数据组文件。"""

    __slots__ = ("condition", "batch_dir", "index", "size", "path",
                 "chambers", "mean_mu", "mu_from_name")

    def __init__(self, condition, batch_dir, index, size, path,
                 chambers, mean_mu):
        self.condition = condition      # 自变量文件夹名（源侧工况）
        self.batch_dir = batch_dir
        self.index = index              # 组序号 i → 对应 pptx 第 i 页
        self.size = size                # 文件名里的腔室数 n
        self.path = path
        self.chambers = chambers        # frozenset((source_file, chamber))
        self.mean_mu = mean_mu
        self.mu_from_name = size

    @property
    def key(self):
        return frozenset(self.chambers)

    def __repr__(self):
        return f"<Group {self.condition} 组{self.index}(n={self.size})>"


def _pick_batch_dir(cond_dir, batch_name, logger=None):
    """定位 batch 子目录：优先指定名，否则取 18_batch_* 中最新一个。"""
    preferred = cond_dir / batch_name
    if preferred.is_dir():
        return preferred
    candidates = sorted([d for d in cond_dir.iterdir()
                         if d.is_dir() and d.name.startswith("18_batch_")],
                        key=lambda d: d.name)
    if not candidates:
        return None
    if logger:
        logger.warning("工况「%s」下没有 %s，改用 %s",
                       cond_dir.name, batch_name, candidates[-1].name)
    return candidates[-1]


def load_group_index(agg_dir, logger=None, batch_name=None):
    """
    扫描 数据汇总分析结果/*/18_batch_*/selected__groups_*.xlsx，返回 GroupEntry 列表。
    源目录只读：全程不写入。
    """
    import config

    batch_name = batch_name or config.BATCH_DIR_NAME
    agg_dir = Path(agg_dir)
    entries = []
    for cond_dir in sorted(p for p in agg_dir.iterdir() if p.is_dir()):
        batch_dir = _pick_batch_dir(cond_dir, batch_name, logger)
        if batch_dir is None:
            if logger:
                logger.warning("工况「%s」下找不到 18_batch_* 目录，跳过", cond_dir.name)
            continue
        group_files = sorted(batch_dir.glob(config.GROUP_FILE_PATTERN))
        if not group_files:
            if logger:
                logger.warning("工况「%s」的 %s 下没有 %s",
                               cond_dir.name, batch_dir.name,
                               config.GROUP_FILE_PATTERN)
            continue
        for f in group_files:
            m = GROUP_RE.match(f.name)
            if not m:
                if logger:
                    logger.warning("分组文件名无法解析，跳过：%s", f.name)
                continue
            index, size = int(m.group(1)), int(m.group(2))
            chambers, mean_mu, actual = group_signature(f)
            if actual != size and logger:
                logger.warning("%s：文件名标记 %d 个腔室，实际读出 %d 个",
                               f.name, size, actual)
            entries.append(GroupEntry(cond_dir.name, batch_dir, index, size,
                                      f, frozenset(chambers), mean_mu))
    if logger:
        logger.info("扫到 %d 个数据组文件，分布在 %d 个工况",
                    len(entries), len({e.condition for e in entries}))
    return entries


def match_groups(entry_key, entries):
    """按腔室集合精确匹配。"""
    return [e for e in entries if e.key == entry_key]


def match_groups_by_mu(signature, entries, tol):
    """回退匹配：(腔室数, μmax 均值)。"""
    if signature is None:
        return []
    n, mu = signature
    return [e for e in entries
            if e.size == n and e.mean_mu is not None
            and abs(e.mean_mu - mu) <= tol]
