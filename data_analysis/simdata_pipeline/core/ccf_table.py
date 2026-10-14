# -*- coding: utf-8 -*-
"""
读取 CCF 映射表（ccf_chlorella 筛选出的 20 组）
==============================================
表结构（μmax拟合-20 Groups.xlsx 的 CCF 页，第 1 行为表头）：

    光照 | 氨氮 | 无机碳 | μmax | 原始数据来源

- 前三列：该组在响应曲面中的**自变量组合**（编码 -1/0/1），决定输出文件夹名。
- 第四列：该组的目标 μmax。
- 第五列「原始数据来源」：该 μmax 实际取自原始汇总表的哪个条件、第几个 Response。
  格式形如  '-11-1 res1' / '000 第四行 res1' / '00-1 res3'。

  ⚠️ 注意：自变量组合与来源条件经常**不一致**——μmax 是跨条件挑选后重新配对的，
  因此取原始数据时必须按「原始数据来源」定位，而文件夹名按自变量组合命名。

解析规则
--------
1. 编码部分：'-11-1' → [-1, 1, -1]，用正则 `-?1|0` 切分（'-1'、'0'、'1' 三种 token）。
2. 可选「第X行」：中心点 (0,0,0) 在原始表中有多行，用它区分行号。
3. 'resN'：原始表中第 N 个 Response 列（Response N 的 μmax 均值，
   实测等价于 batch_grouping 在该工况下选出的第 N 组的组均值）。
"""
import re

# 数值可能是 str('−1') 或 int(-1)，'−' 是 U+2212 减号
_MINUS = {"−": "-", "－": "-", "–": "-"}


def _norm_token(s):
    for k, v in _MINUS.items():
        s = s.replace(k, v)
    return s


def _to_int(s):
    return int(_norm_token(str(s)).strip())


def parse_source(src: str):
    """
    解析「原始数据来源」字符串。
    返回 dict：code=(光照,氨氮,无机碳)、row_hint=第几行（或 None）、res=Response 序号（或 None）
    """
    if src is None:
        return None
    s = _norm_token(str(src)).strip()

    # resN
    m_res = re.search(r"res\s*(\d+)", s)
    res = int(m_res.group(1)) if m_res else None

    # 第X行
    m_row = re.search(r"第\s*(.+?)\s*行", s)
    row_hint = m_row.group(1) if m_row else None

    # 编码部分：取 res / 第X行 之前的片段
    code_part = re.split(r"res\s*\d+|第\s*.+?\s*行", s)[0].strip()
    tokens = re.findall(r"-?1|0", code_part)
    if len(tokens) != 3:
        return {"code": None, "row_hint": row_hint, "res": res, "raw": src}
    code = tuple(int(t) for t in tokens)
    return {"code": code, "row_hint": row_hint, "res": res, "raw": src}


def load_ccf_rows(xlsx_path, sheet_name=None, logger=None):
    """
    读取 CCF 页，返回 20 行的列表，每项：
        {idx, design_code, mu_max, source_raw, source}
    """
    from pathlib import Path
    import openpyxl
    import config

    xlsx_path = Path(xlsx_path)
    sheet_name = sheet_name or config.CCF_SHEET_NAME
    if not xlsx_path.exists():
        raise FileNotFoundError(f"CCF 映射表不存在: {xlsx_path}")

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise KeyError(f"工作表 '{sheet_name}' 不存在，实际: {wb.sheetnames}")
    ws = wb[sheet_name]

    rows = []
    for i, r in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if r is None or r[0] is None:
            continue
        design_code = tuple(_to_int(r[c]) for c in (0, 1, 2))
        mu = r[3]
        src = r[4]
        rows.append({
            "row": i,
            "design_code": design_code,
            "mu_max": float(mu) if mu is not None else None,
            "source_raw": src,
            "source": parse_source(src),
        })

    if logger:
        bad = [x for x in rows if x["source"]["code"] is None]
        if bad:
            logger.warning("CCF 表中有 %d 行的「原始数据来源」无法解析编码: %s",
                           len(bad), [x["source_raw"] for x in bad])
        logger.info("CCF 映射表读取完成：%d 行", len(rows))
    return rows
