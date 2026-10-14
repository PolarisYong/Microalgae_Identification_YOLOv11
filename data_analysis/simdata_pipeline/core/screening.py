# -*- coding: utf-8 -*-
"""
按微藻生长规律筛查异常腔室
==========================
在二次分组之前，把明显不符合微藻典型生长规律（延滞 → 对数 → 稳定，末期平缓）
的腔室剔除，避免污染分组结果。

判定用「目标数量」列（相对细胞数量，已按 0 h 归一化）的时间序列，逐条给出
可读的中文原因，原因会写进「剔除腔室数据.xlsx」的对应页签。
"""
from __future__ import annotations

import math

import config


# --------------------------------------------------------------------------- 序列特征
def _clean(values):
    """去空值 / NaN / inf，返回 float 列表。"""
    out = []
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _moving_median(values, window=5):
    """居中移动中位数；两端用可用窗口补齐。"""
    n = len(values)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(values[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def longest_decline_run(values):
    """最长连续下降（严格递减）段的长度。"""
    best = run = 0
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best + 1 if best else 1


def profile(counts, areas=None):
    """从一条生长序列里提取全部判定特征。"""
    c = _clean(counts)
    n = len(c)
    if n == 0:
        return None
    start = c[0]
    peak = max(c)
    peak_at = c.index(peak)
    tail = c[-3:] if n >= 3 else c[-1:]
    end = sum(tail) / len(tail)
    smooth = _moving_median(c, 5)
    resid = [abs(c[i] - smooth[i]) / smooth[i] for i in range(n) if smooth[i] > 0]
    resid.sort()
    noise = resid[len(resid) // 2] if resid else 0.0
    jumps = 0
    for i in range(n - 1):
        base = max(abs(c[i]), 1e-9)
        if abs(c[i + 1] - c[i]) / base > config.SCREEN_JUMP_RATIO:
            jumps += 1
    return {
        "n": n,
        "start": start,
        "peak": peak,
        "peak_at": peak_at,
        "end": end,
        "fold": peak / start if start > 0 else float("inf"),
        "end_peak": end / peak if peak > 0 else 0.0,
        "decline_run_ratio": longest_decline_run(c) / n,
        "noise": noise,
        "jumps": jumps,
        "nonpositive": sum(1 for v in c if v <= 0),
    }


# --------------------------------------------------------------------------- 判定
def evaluate(ref) -> str:
    """
    判定一个腔室是否异常。

    返回 None 表示正常；否则返回中文原因串。
    """
    if not config.SCREEN_ENABLE:
        return None

    total_rows = max(len(ref.rows) - 1, 0)          # 去掉表头
    c = _clean(ref.counts)
    n = len(c)

    # ① 数据缺失
    if n < config.SCREEN_MIN_POINTS:
        return f"数据缺失：有效数据点仅 {n} 个（阈值 {config.SCREEN_MIN_POINTS}）"
    if total_rows and (total_rows - n) / total_rows > config.SCREEN_MAX_MISSING_RATIO:
        missing = (total_rows - n) / total_rows * 100
        return f"数据缺失：目标数量列缺失 {missing:.1f}%（阈值 {config.SCREEN_MAX_MISSING_RATIO * 100:.0f}%）"
    if total_rows and ref.fail_count / total_rows > config.SCREEN_MAX_FAIL_RATIO:
        return f"数据缺失：处理状态非「成功」{ref.fail_count} 行 / 共 {total_rows} 行"

    p = profile(ref.counts)
    if p is None:
        return "数据缺失：无法解析任何有效数据点"

    # ② 非正值（无法进入 ln 尺度拟合）
    if p["nonpositive"]:
        return f"含非正值数据点：{p['nonpositive']} 个"

    # ③ 无生长
    if p["fold"] < config.SCREEN_MIN_GROWTH_FOLD:
        return f"无生长：峰值仅为初始的 {p['fold']:.2f} 倍（阈值 {config.SCREEN_MIN_GROWTH_FOLD}）"

    # ④ 长期下降（衰亡）
    if p["end_peak"] < config.SCREEN_MIN_END_PEAK_RATIO:
        return f"长期下降：末期降至峰值的 {p['end_peak'] * 100:.1f}%（阈值 {config.SCREEN_MIN_END_PEAK_RATIO * 100:.0f}%）"
    if p["decline_run_ratio"] > config.SCREEN_MAX_DECLINE_RUN_RATIO:
        return (f"长期下降：最长连续下降段占序列 "
                f"{p['decline_run_ratio'] * 100:.1f}%（阈值 {config.SCREEN_MAX_DECLINE_RUN_RATIO * 100:.0f}%）")

    # ⑤ 剧烈非典型波动
    if p["jumps"] > config.SCREEN_MAX_JUMP_COUNT:
        return (f"剧烈非典型波动：相邻点相对变化 >"
                f"{config.SCREEN_JUMP_RATIO * 100:.0f}% 的突跳 {p['jumps']} 次"
                f"（阈值 {config.SCREEN_MAX_JUMP_COUNT}）")
    if p["noise"] > config.SCREEN_MAX_NOISE_RATIO:
        return (f"剧烈非典型波动：相对移动中位数残差中位数 {p['noise'] * 100:.1f}%"
                f"（阈值 {config.SCREEN_MAX_NOISE_RATIO * 100:.0f}%）")
    return None


def poor_fit_reason(ref) -> str:
    """
    batch_grouping 的严格候选条件（N0 > 0 且 R² > 0.9）不满足时的原因串。
    返回 None 表示通过。
    """
    if not config.SCREEN_DROP_POOR_FIT:
        return None
    if not (ref.n0 > config.STRICT_MIN_N0):
        return f"拟合起始相对数量 Nrel(0) ≤ 0（{ref.n0}），不满足 batch_grouping 严格候选条件"
    if not (ref.r2 > config.STRICT_MIN_R2):
        return f"拟合优度 R² = {ref.r2:.4f} ≤ {config.STRICT_MIN_R2}，不满足 batch_grouping 严格候选条件"
    return None


def partition(refs, logger=None, tag=""):
    """
    把腔室分成 (保留, 剔除)。

    返回 (kept, excluded)
        kept     : list[ChamberRef]
        excluded : list[(ChamberRef, 原因)]
    """
    kept, excluded = [], []
    for ref in refs:
        reason = evaluate(ref)
        if reason is None:
            reason = poor_fit_reason(ref)
        if reason is None:
            kept.append(ref)
        else:
            excluded.append((ref, reason))
            if logger:
                logger.info("%s 剔除 %s｜%s —— %s", tag, ref.condition,
                            ref.sheet, reason)
    if logger:
        logger.info("%s 筛查：保留 %d 个，剔除 %d 个", tag, len(kept), len(excluded))
    return kept, excluded
