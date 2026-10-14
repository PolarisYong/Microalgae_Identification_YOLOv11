# -*- coding: utf-8 -*-
"""
单元格富文本（斜体 / 上下标）构建
==================================
规则逐 run 复刻自参考模板「可视化结果参考模板.xlsx」（rich_text=True 读取）：

- 变量字母斜体：μ、K、N、T、t、λ、R、Y、I、C、F、η …
- 描述性下标：正体下标（max、rel、rel,fit、lag、prolif、plat、d、90、N、DIC、0）
- 单位指数：上标（h 的 -1、m 的 -2、s 的 -1、L 的 -1）
- R² 用 Unicode ²（模板即如此，非上标格式）
- NH₄⁺ 用 Unicode ₄⁺

spec 是一串 (text, style)，style ∈ {"", "i", "sub", "sup"}。
所有 spec 的文本拼接与 en_labels 中原纯字符串逐字一致（仅补格式，不改文字）。
"""
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont

FONT_NAME = "Times New Roman"


def rt(spec, bold=False, size=11):
    """spec → CellRichText，全部 run 统一 Times New Roman。"""
    blocks = []
    for text, style in spec:
        vert = {"sub": "subscript", "sup": "superscript"}.get(style)
        blocks.append(TextBlock(
            InlineFont(rFont=FONT_NAME, sz=size, b=bold,
                       i=(style == "i"), vertAlign=vert),
            text))
    return CellRichText(*blocks)


def set_rt(cell, spec, bold=False, size=11):
    """把富文本写入单元格；spec 为 str 时写纯文本，None 时跳过。"""
    if spec is None:
        return
    cell.value = spec if isinstance(spec, str) else rt(spec, bold=bold, size=size)


# --------------------------------------------------------------------- 参数名 spec
MU_MAX = [("Maximum specific growth rate, ", ""), ("μ", "i"), ("max", "sub"),
          (" (h", ""), ("-1", "sup"), (")", "")]
K_REL = [("Relative late-stage cell-population level, ", ""), ("K", "i"),
         ("rel ", "sub"), ("(dimensionless)", "")]
N_REL_FIT = [("Fitted initial relative cell number, ", ""), ("N", "i"),
             ("rel,fit", "sub"), ("(0) (dimensionless)", "")]
LAMBDA = [("Lag-phase parameter, ", ""), ("λ ", "i"), ("(h)", "")]
R2_SCALE = [("Goodness of fit, ", ""), ("R", "i"), ("² (", ""), ("Y", "i"),
            (" = ln(", ""), ("N", "i"), ("rel", "sub"), (") scale) ", "")]
R2_PLAIN = [("Goodness of fit, ", ""), ("R", "i"), ("² ", "")]
T_D = [("Cell-number doubling time, ", ""), ("T", "i"), ("d ", "sub"), ("(h)", "")]
T_LAG_FULL = [("Lag-phase duration, ", ""), ("T", "i"), ("lag", "sub"),
              (" (", ""), ("T", "i"), ("lag", "sub"), (" = ", ""), ("λ", "i"),
              (") (h)", "")]
T_LAG_SHORT = [("Lag-phase duration, ", ""), ("T", "i"), ("lag", "sub"), (" (h)", "")]
T_PROLIF = [("Rapid proliferation phase duration, ", ""), ("T", "i"),
            ("prolif", "sub"), (" (h)", "")]
T_PLAT = [("Plateau-phase duration, ", ""), ("T", "i"), ("plat", "sub"), (" (h)", "")]
T90 = [("Plateau-phase onset time, ", ""), ("t", "i"), ("90 ", "sub"), ("(h)", "")]

# --------------------------------------------------------------------- 培养条件块
COND_LIGHT = [("Light intensity, ", ""), ("I", "i"), (" (μmol m", ""),
              ("-2", "sup"), (" s", ""), ("-1", "sup"), (")", "")]
COND_N = [("NH₄⁺-N concentration, ", ""), ("C", "i"), ("N", "sub"),
          (" (mg N L", ""), ("-1", "sup"), (")", "")]
COND_DIC = [("DIC concentration, ", ""), ("C", "i"), ("DIC", "sub"),
            (" (mg C L", ""), ("-1", "sup"), (")", "")]

# --------------------------------------------------------------------- 列表（与 en_labels 对齐）
# 汇总结果页表①表头（bold, sz 11, 行高 44）
SUMMARY_HEADER_SPECS = ["Valid cultivation unit ID", MU_MAX, K_REL, T_LAG_FULL,
                        T_PROLIF, T_PLAT, T_D, R2_PLAIN]
# 培养条件块表头行（模板：非 bold, sz 10）
COND_BLOCK_SPECS = [COND_LIGHT, COND_N, COND_DIC]
# Pooled fitting results 参数名（非 bold, sz 11；前两行是纯文本）
POOLED_ROW_SPECS = [None, None, MU_MAX, K_REL, N_REL_FIT, LAMBDA, R2_SCALE,
                    T_LAG_FULL, T_PROLIF, T_PLAT, T90]
# 腔室页 E 列参数名（非 bold, sz 11）
CHAMBER_PARAM_SPECS = [MU_MAX, K_REL, N_REL_FIT, LAMBDA, R2_SCALE, T_D,
                       T_LAG_FULL, T_PROLIF, T_PLAT, T90]

# --------------------------------------------------------------------- selected 表头
# 键 = en_labels.SELECTED_HEADER_MAP 的英文译文；符号按变量斜体/下标正体处理
SELECTED_HEADER_SPECS = {
    "Maximum specific growth rate, μmax (h-1)": MU_MAX,
    "Carrying capacity, K (cells/chamber)":
        [("Carrying capacity, ", ""), ("K", "i"), (" (cells/chamber)", "")],
    "Initial cell number, N0 (cells/chamber)":
        [("Initial cell number, ", ""), ("N", "i"), ("0", "sub"),
         (" (cells/chamber)", "")],
    "Goodness of fit, R²":
        [("Goodness of fit, ", ""), ("R", "i"), ("²", "")],
    "Cell-number doubling time, Td (h)": T_D,
    "Proliferation fold, F":
        [("Proliferation fold, ", ""), ("F", "i")],
    "Growth efficiency, η (1/F)":
        [("Growth efficiency, ", ""), ("η", "i"), (" (1/", ""), ("F", "i"), (")", "")],
    "Lag-phase duration, Tlag (h)": T_LAG_SHORT,
    "Rapid proliferation phase duration, Tprolif (h)": T_PROLIF,
    "Plateau-phase duration, Tplat (h)": T_PLAT,
}
