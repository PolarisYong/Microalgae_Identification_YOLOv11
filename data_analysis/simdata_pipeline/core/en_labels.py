# -*- coding: utf-8 -*-
"""
英文版标签与条件映射
====================
供 build_english_version.py 使用，集中管理：

1. 文件夹名 → 培养条件（coded 三元组、L/N/DIC 实际值、图标题用条件标签）
   - 文件夹名形如 L120‑N160‑IC5.25%（U+2011 连字符），IC 百分比 ↔ DIC mg C/L：
       IC0.5%  ↔ DIC75  （编码 -1）
       IC5.25% ↔ DIC320 （编码  0）
       IC10%   ↔ DIC565 （编码 +1）
     编码定义见 ccf_chlorella/ccf_from_table.py：
       光照 -1/0/+1 = 30/120/210 μmol/m²/s（中心 120，半幅 90）
       氨氮 -1/0/+1 = 20/160/300 mg/L      （中心 160，半幅 140）
       无机碳 -1/0/+1 = 75/320/565 mg C/L  （中心 320，半幅 245）
   - 最优模拟组文件夹「2208」固定为 L182-N287-DIC461（CCF 驻点，
     光照=182、氨氮=287、无机碳=461，由用户给定）。
2. 英文模板（可视化结果参考模板.xlsx）的全部固定字符串。
3. selected__groups_*.xlsx 表头的中→英映射。
"""
import re

# --------------------------------------------------------------------- 条件映射
# IC 百分比字符串 → DIC 实际值（mg C/L）
IC_TO_DIC = {"0.5": 75, "5.25": 320, "10": 565}

# 最优模拟组（文件夹 2208）的条件：光照=182 μmol/m²/s，氨氮=287 mg/L，无机碳=461 mg C/L
OPTIMAL_FOLDER = "2208"
OPTIMAL_CONDITION = {"L": 182, "N": 287, "DIC": 461}

# 文件夹名解析：L120‑N160‑IC5.25%（‑ 为 U+2011，也兼容普通连字符）
FOLDER_COND_RE = re.compile(
    r"^L(\d+)[\u2011-]N(\d+)[\u2011-]IC([\d.]+)%$")


def _coded(value, center, half_width):
    """实际值 → CCF 编码；正好落在 -1/0/+1 上时返回 int，否则保留两位小数。"""
    code = (value - center) / half_width
    if abs(code - round(code)) < 1e-9 and round(code) in (-1, 0, 1):
        return int(round(code))
    return round(code, 2)


def condition_of_folder(folder_name):
    """
    文件夹名 → dict(L, N, DIC, coded, label, title_label)
        L/N/DIC     : 实际值
        coded       : (x1, x2, x3) 编码三元组
        label       : 条件取值标签，如 "L120-N160-DIC320"（培养条件块第 3 行用）
        title_label : 图标题里的条件，如 "L120-N160-DIC320"
        coded_text  : "(-1, -1, -1)"（培养条件块标题用）
    无法识别时返回 None。
    """
    if folder_name == OPTIMAL_FOLDER:
        values = dict(OPTIMAL_CONDITION)
    else:
        m = FOLDER_COND_RE.match(folder_name)
        if not m:
            return None
        light, nitrogen, ic = int(m.group(1)), int(m.group(2)), m.group(3)
        if ic not in IC_TO_DIC:
            return None
        values = {"L": light, "N": nitrogen, "DIC": IC_TO_DIC[ic]}

    coded = (_coded(values["L"], 120, 90),
             _coded(values["N"], 160, 140),
             _coded(values["DIC"], 320, 245))
    label = f"L{values['L']}-N{values['N']}-DIC{values['DIC']}"
    return {
        **values,
        "coded": coded,
        "coded_text": f"({coded[0]}, {coded[1]}, {coded[2]})",
        "label": label,
        "title_label": label,
        "values": (f"L{values['L']}", f"N{values['N']}", f"DIC{values['DIC']}"),
    }


# --------------------------------------------------------------------- 汇总结果页
SUMMARY_SHEET_EN = "汇总结果"          # 模板页签名（模板本身即为中英文混合命名）

# 表① 腔室级汇总表头（与模板逐字一致）
SUMMARY_HEADERS = [
    "Valid cultivation unit ID",
    "Maximum specific growth rate, μmax (h-1)",
    "Relative late-stage cell-population level, Krel (dimensionless)",
    "Lag-phase duration, Tlag (Tlag = λ) (h)",
    "Rapid proliferation phase duration, Tprolif (h)",
    "Plateau-phase duration, Tplat (h)",
    "Cell-number doubling time, Td (h)",
    "Goodness of fit, R² ",
]

# 培养条件块（模板第 22-24 行；"for the independent run" 不适用于 CCF 设计点，略去）
COND_BLOCK_TITLE = "Cultivation conditions {coded}"       # coded → "(-1, -1, -1)"
COND_BLOCK_HEADERS = [
    "Light intensity, I (μmol m-2 s-1)",
    "NH₄⁺-N concentration, CN (mg N L-1)",
    "DIC concentration, CDIC (mg C L-1)",
]

# 表③ 合并拟合结果（模板第 26-37 行）
POOLED_TITLE = "Pooled fitting results"
POOLED_VALUE_HEADER = "Value"
POOLED_ROWS = [
    "Number of worksheets included in fitting",
    "Total number of data points",
    "Maximum specific growth rate, μmax (h-1)",
    "Relative late-stage cell-population level, Krel (dimensionless)",
    "Fitted initial relative cell number, Nrel,fit(0) (dimensionless)",
    "Lag-phase parameter, λ (h)",
    "Goodness of fit, R² (Y = ln(Nrel) scale) ",
    "Lag-phase duration, Tlag (Tlag = λ) (h)",
    "Rapid proliferation phase duration, Tprolif (h)",
    "Plateau-phase duration, Tplat (h)",
    "Plateau-phase onset time, t90 (h)",
]

# --------------------------------------------------------------------- 腔室页
CHAMBER_HEADERS = [
    "Image filename", "Processing status", "Cell count",
    "Total cell projected area (μm²)", "Parameter", "Value",
]
# E/F 参数块 10 行（与模板 CB-1 逐字一致）
CHAMBER_PARAM_LABELS = [
    "Maximum specific growth rate, μmax (h-1)",
    "Relative late-stage cell-population level, Krel (dimensionless)",
    "Fitted initial relative cell number, Nrel,fit(0) (dimensionless)",
    "Lag-phase parameter, λ (h)",
    "Goodness of fit, R² (Y = ln(Nrel) scale) ",
    "Cell-number doubling time, Td (h)",
    "Lag-phase duration, Tlag (Tlag = λ) (h)",
    "Rapid proliferation phase duration, Tprolif (h)",
    "Plateau-phase duration, Tplat (h)",
    "Plateau-phase onset time, t90 (h)",
]
STATUS_EN = {"成功": "Success", "失败": "Failed"}

# --------------------------------------------------------------------- 中文源参数键 → 规范键
# 腔室页 E 列 / 汇总页合并块的中文标签（子串匹配）→ 规范键
CN_PARAM_KEYS = [
    ("mu",    ("μmax", "umax")),      # 部分源文件写作 ASCII 'uMAX'
    ("K",     ("Krel",)),
    ("N0",    ("Nrel(0)",)),
    ("lam",   ("λ",)),
    ("r2",    ("R²",)),
    ("td",    ("T_d",)),
    ("tlag",  ("滞后期时长",)),
    ("tprolif", ("对数期时长",)),
    ("tplat", ("稳定期时长",)),
    ("t90",   ("t90",)),
]
# 汇总页合并块专有
CN_MERGED_KEYS = [
    ("n_worksheets", ("参与拟合的页签数量",)),
    ("n_points",     ("总数据点数量",)),
    ("mu",    ("μmax", "umax")),
    ("K",     ("Krel",)),
    ("N0",    ("Nrel(0)",)),
    ("lam",   ("λ",)),
    ("r2",    ("R²",)),
    ("tlag",  ("滞后期时长",)),
    ("tprolif", ("对数期时长",)),
    ("tplat", ("稳定期时长",)),
    ("t90",   ("t90",)),
]


def _normalize_mu(text):
    """μ 有两种 Unicode 写法：U+03BC (greek mu) 与 U+00B5 (micro sign)。
    汇总页标签用前者、腔室页 E 列沿用原始文件的后者，统一后再匹配。"""
    return str(text).replace("µ", "μ")


def map_params(cn_params, key_spec=CN_PARAM_KEYS):
    """{中文标签: 值} → {规范键: 值}（按子串匹配，未识别的忽略）。"""
    out = {}
    for label, value in (cn_params or {}).items():
        text = _normalize_mu(label).lower()
        for key, needles in key_spec:
            if key in out:
                continue
            if any(_normalize_mu(n).lower() in text for n in needles):
                out[key] = value
                break
    return out


# --------------------------------------------------------------------- selected 文件表头
SELECTED_HEADER_MAP = {
    "源文件名称(source_file)": "Source file",
    "微流道编号(flow)": "Flow channel",
    "腔室名称(chamber)": "Chamber",
    "最大比生长速率μmax (h^-1)": "Maximum specific growth rate, μmax (h-1)",
    "环境容纳量K (个/腔室)": "Carrying capacity, K (cells/chamber)",
    "初始细胞数量N0 (个/腔室)": "Initial cell number, N0 (cells/chamber)",
    "拟合优度R²": "Goodness of fit, R²",
    "平均细胞周期T_d(h)": "Cell-number doubling time, Td (h)",
    "增殖倍数 F": "Proliferation fold, F",
    "生长效率 η (1/增殖倍数)": "Growth efficiency, η (1/F)",
    "滞后期时长(h)": "Lag-phase duration, Tlag (h)",
    "对数期时长(h)": "Rapid proliferation phase duration, Tprolif (h)",
    "稳定期时长(h)": "Plateau-phase duration, Tplat (h)",
}

# --------------------------------------------------------------------- PPT 页眉
PPT_GROUP_RE = re.compile(r"数据组\s*(\d+)\s*[（(]\s*n\s*=\s*(\d+)\s*[)）]")


def translate_ppt_header(text):
    """「数据组 1（n = 18）」→「Group 1 (n = 18)」，其余文本原样返回。"""
    return PPT_GROUP_RE.sub(lambda m: f"Group {m.group(1)} (n = {m.group(2)})", text)
