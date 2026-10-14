# -*- coding: utf-8 -*-
"""
工况（自变量组合）的编码 ↔ 目录名 互转
=====================================
CCF 三因子各取 -1 / 0 / +1 三个编码水平：

    光照强度   -1 → L30    0 → L120    1 → L210    (μmol/m²/s)
    氨氮浓度   -1 → N20    0 → N160    1 → N300    (mg/L)
    无机碳     -1 → IC0.5% 0 → IC5.25% 1 → IC10%   (CO2 浓度)

目录名形如「L30‑N20‑IC10%」，其中的连字符是 Unicode U+2011，
与 ASCII '-' 不同，匹配时必须用 config.COND_SEP。
"""
import re
import config


def code_to_name(light: int, nitrogen: int, ic: int) -> str:
    """编码三元组 → 工况目录名，如 (1, -1, -1) → 'L210‑N20‑IC0.5%'"""
    return config.COND_SEP.join([
        config.LEVEL_LIGHT[int(light)],
        config.LEVEL_NITROGEN[int(nitrogen)],
        config.LEVEL_IC[int(ic)],
    ])


def name_to_code(name: str):
    """工况目录名 → 编码三元组；解析失败返回 None。"""
    # 同时容忍 U+2011 与 ASCII '-'
    m = re.match(r"^(L\d+)[\-\u2010-\u2015](N\d+)[\-\u2010-\u2015](IC[\d.]+%)$",
                 name.strip())
    if not m:
        return None
    l = config.NAME_TO_LEVEL_LIGHT.get(m.group(1))
    n = config.NAME_TO_LEVEL_NITROGEN.get(m.group(2))
    c = config.NAME_TO_LEVEL_IC.get(m.group(3))
    if None in (l, n, c):
        return None
    return (l, n, c)


def list_condition_dirs(agg_dir):
    """
    扫描「数据汇总分析结果」下的工况目录。
    返回 {编码三元组: {'name': 目录名, 'path': 绝对路径}}
    无法解析的目录名会被跳过并记录到返回值的 '__skipped__'。
    """
    from pathlib import Path
    agg_dir = Path(agg_dir)
    result = {}
    skipped = []
    if not agg_dir.exists():
        return result, [f"目录不存在: {agg_dir}"]
    for d in sorted(agg_dir.iterdir()):
        if not d.is_dir():
            continue
        code = name_to_code(d.name)
        if code is None:
            skipped.append(d.name)
            continue
        result[code] = {"name": d.name, "path": d}
    return result, skipped
