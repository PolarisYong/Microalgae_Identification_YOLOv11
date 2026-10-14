# -*- coding: utf-8 -*-
"""
合并拟合（修正 Logistic，Y = ln(Nrel) 尺度）
==========================================
移植自参考实现 `08_processed_batch.py`，供「汇总」页签的第三张表使用。

与参考实现的唯一差别：**不做腔室筛选**。
参考实现会用后向剔除（backward elimination）+ beam search + 回看修复，
按 R²≥0.8 的目标挑一个「最优腔室子集」再拟合；本模块按需求跳过这一步，
把所有数据可用的腔室直接拼接（沿时间轴首尾相接）后一次性拟合。

模型（Zwietering 修正 Logistic，在 Y = ln(Nrel) 尺度上拟合）：

    Y(t)    = A / (1 + exp(4μ/A·(λ − t) + 2))
    Nrel(t) = exp(Y(t))

参数含义：
    μ       —— 最大比生长速率 μmax (h⁻¹)，即 Y(t) 曲线的最大斜率
    A       —— ln 尺度的平台高度，Krel = exp(A) 为相对平台倍数（无量纲）
    λ       —— 滞止期参数 (h)
    Nrel(0) —— Nrel 在 t_min 处的拟合值，起始相对数量（无量纲）
    R²      —— 在 Y = ln(Nrel) 尺度上计算（与参考实现口径一致；
               不能与 Nrel 尺度的 R² 混用）
"""
import numpy as np
from scipy.optimize import curve_fit

import config


# --------------------------------------------------------------------------- 模型
def fit_model(t, mu_fit, amplitude, lambda_):
    """Y(t) = A / (1 + exp(4μ/A·(λ − t) + 2))，拟合时使用的响应变量形式。"""
    t = np.asarray(t, dtype=float)
    if not np.isfinite(mu_fit) or not np.isfinite(amplitude) or not np.isfinite(lambda_):
        return np.full_like(t, np.nan, dtype=float)
    amplitude = float(amplitude)
    if amplitude <= 0:
        return np.full_like(t, np.nan, dtype=float)
    mu_fit = max(float(mu_fit), 1e-12)
    exponent = (4.0 * mu_fit / amplitude) * (float(lambda_) - t) + 2.0
    exponent = np.clip(exponent, -700, 700)
    return amplitude / (1.0 + np.exp(exponent))


def model(t, mu_fit, krel, lambda_):
    """Nrel(t) = exp(Y(t))，返回相对数量尺度上的曲线。"""
    t = np.asarray(t, dtype=float)
    krel = float(krel)
    if not np.isfinite(krel) or krel <= 1.0:
        return np.full_like(t, np.nan, dtype=float)
    return np.exp(fit_model(t, mu_fit, np.log(krel), lambda_))


def time_at_fraction(p, mu_fit, krel, lambda_):
    """曲线达到平台倍数比例 p 的时刻（h）。p=0.90 即参考实现的 t90。"""
    if not np.isfinite(mu_fit) or not np.isfinite(krel) or not np.isfinite(lambda_):
        return np.nan
    if p <= 0 or p >= 1:
        return np.nan
    amplitude = np.log(float(krel)) if krel > 1.0 else np.nan
    if not np.isfinite(amplitude) or amplitude <= 0 or mu_fit <= 0:
        return float(lambda_)
    return float(lambda_) + amplitude / (4.0 * float(mu_fit)) * (2.0 + np.log(p / (1.0 - p)))


def growth_phases(t, mu_fit, krel, lambda_, stable_fraction=None):
    """
    计算滞后期 / 对数期 / 稳定期时长（h）。
    返回 (phase_duration, phases)，前者是 {字段名: 时长} 字典。
    """
    stable_fraction = (config.MERGE_STABLE_FRACTION
                       if stable_fraction is None else stable_fraction)
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return {"滞后期时长(h)": 0, "对数期时长(h)": 0, "稳定期时长(h)": 0}, []

    t_start = float(t[0])
    t_end = float(t[-1])
    lag_end = float(lambda_) if np.isfinite(lambda_) else t_start
    stable_start = time_at_fraction(stable_fraction, mu_fit, krel, lambda_)
    if not np.isfinite(stable_start):
        stable_start = lag_end
    stable_start = max(stable_start, lag_end)

    lag_duration = max(min(lag_end, t_end) - t_start, 0.0)
    log_duration = max(min(stable_start, t_end) - max(lag_end, t_start), 0.0)
    stable_duration = max(t_end - max(stable_start, t_start), 0.0)

    phase_duration = {
        "滞后期时长(h)": round(lag_duration, 2),
        "对数期时长(h)": round(log_duration, 2),
        "稳定期时长(h)": round(stable_duration, 2),
    }
    phases = [
        {"阶段": "滞后期", "开始时间(h)": t_start,
         "结束时间(h)": min(max(lag_end, t_start), t_end), "时长(h)": lag_duration},
        {"阶段": "对数期", "开始时间(h)": min(max(lag_end, t_start), t_end),
         "结束时间(h)": min(max(stable_start, t_start), t_end) if stable_start > lag_end
                        else min(max(lag_end, t_start), t_end),
         "时长(h)": log_duration},
        {"阶段": "稳定期", "开始时间(h)": min(max(stable_start, t_start), t_end),
         "结束时间(h)": t_end, "时长(h)": stable_duration},
    ]
    return phase_duration, phases


# --------------------------------------------------------------------------- 拟合
def fit_curve(t, counts, mu_guess=0.1, k_guess=None):
    """
    单条曲线拟合。`t` 为时间（h），`counts` 为相对数量 Nrel。

    多组初值依次尝试：修正 Logistic 在 A 尺度上的最小二乘对初值敏感，
    单个初值容易发散。全部失败时抛最后一个异常。
    """
    t = np.asarray(t, dtype=float)
    counts = np.asarray(counts, dtype=float)

    if t.size < 3:
        raise ValueError("数据点不足，无法拟合")
    if not np.isfinite(t).all() or not np.isfinite(counts).all():
        raise ValueError("数据中包含非数值")
    if np.any(counts <= 0):
        raise ValueError("取对数前要求 Nrel 全部为正")

    max_count = float(np.max(counts)) if counts.size else 1.0
    if k_guess is None:
        k_guess = max(max_count, 1.0)

    mu_candidates = [mu_guess, max(mu_guess * 0.5, 1e-4), 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    t_span = max(float(t[-1] - t[0]), 1.0)
    lambda_guess = float(t[min(len(t) - 1, max(1, int(round(len(t) * 0.2))))])
    k_upper = max(max_count * 10.0, float(k_guess) * 5.0, 1.000001)
    amplitude_guess = max(np.log(max(float(k_guess), 1.000001)), 1e-3)
    amplitude_upper = max(np.log(k_upper), 1.0)

    bounds = (
        [1e-6, 1e-6, float(t[0]) - t_span],
        [50.0, amplitude_upper, float(t[-1]) + t_span],
    )

    last_error = None
    for candidate_mu in mu_candidates:
        p0 = np.array([candidate_mu, amplitude_guess, lambda_guess], dtype=float)
        p0 = np.clip(p0, np.array(bounds[0]) + 1e-12, np.array(bounds[1]) - 1e-12)
        try:
            popt, _ = curve_fit(
                f=fit_model, xdata=t, ydata=np.log(counts),
                p0=p0, bounds=bounds, maxfev=20000,
            )
        except Exception as exc:            # noqa: BLE001 —— 逐个初值回退
            last_error = exc
            continue

        mu_fit, amplitude_fit, lambda_fit = popt
        krel_fit = float(np.exp(amplitude_fit))
        y_fit_log = fit_model(t, mu_fit, amplitude_fit, lambda_fit)
        y_fit = np.exp(y_fit_log)
        n0_fit = float(model(np.array([t.min()]), mu_fit, krel_fit, lambda_fit)[0])

        # R² 与参考实现一致：在 Y = ln(Nrel) 尺度上算
        y_obs = np.log(counts)
        ss_res_y = float(np.sum((y_obs - y_fit_log) ** 2))
        ss_tot_y = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
        r2 = 1 - (ss_res_y / ss_tot_y) if ss_tot_y != 0 else np.nan

        # μmax 为 Y(t) 的最大斜率；在参数化下等于 μ_fit
        t_d = np.log(2) / mu_fit if mu_fit > 0 else np.nan

        return {
            "mu_fit": float(mu_fit),
            "mu_max": float(mu_fit),
            "mu_max_t": float(lambda_fit) + amplitude_fit / (2.0 * float(mu_fit)),
            "K": krel_fit,
            "N0": n0_fit,
            "lambda_": float(lambda_fit),
            "r2": float(r2),
            "t_d": float(t_d),
            "y_fit": y_fit,
            "y_fit_log": y_fit_log,
        }

    raise last_error if last_error is not None else RuntimeError("拟合失败")


# --------------------------------------------------------------------------- 合并拟合
def merged_fit(named_series, min_points=None, stable_fraction=None):
    """
    把多个腔室的 Nrel 序列沿时间轴拼接后一次性拟合（不做腔室筛选）。

    参数
    ----
    named_series : list[(腔室名, Nrel 序列)]
        每个序列取自腔室页签的「目标数量」列（相对数量，首点为 1）。

    返回
    ----
    dict:
        valid_sheets   —— 实际参与拟合的腔室名（按输入顺序）
        excluded       —— 未参与的腔室名（数据不足 / 含非正值 / 非数值）
        fit            —— 拟合结果字典；腔室不足 2 个或拟合失败时为 None
        merged_t       —— 拼接后的时间向量
        merged_counts  —— 拼接后的相对数量
        phase_duration —— {滞后期/对数期/稳定期时长(h)}
        t90            —— 稳定期起始时刻 (h)
    """
    min_points = config.MERGE_MIN_POINTS if min_points is None else min_points
    stable_fraction = (config.MERGE_STABLE_FRACTION
                       if stable_fraction is None else stable_fraction)

    valid, excluded = [], []
    for name, series in named_series:
        arr = np.asarray([v for v in (series or []) if v is not None], dtype=float)
        if arr.size < min_points or not np.isfinite(arr).all() or np.any(arr <= 0):
            excluded.append(name)
            continue
        valid.append((name, arr))

    result = {
        "valid_sheets": [name for name, _ in valid],
        "excluded": excluded,
        "fit": None,
        "merged_t": None,
        "merged_counts": None,
        "phase_duration": None,
        "t90": np.nan,
    }
    if len(valid) < 2:
        return result

    merged_t = np.concatenate([np.arange(len(arr), dtype=float) for _, arr in valid])
    merged_counts = np.concatenate([arr for _, arr in valid])

    try:
        fit = fit_curve(merged_t, merged_counts,
                        mu_guess=0.1, k_guess=float(np.max(merged_counts)))
    except Exception:                       # noqa: BLE001 —— 拟合失败不阻断写出
        return result

    # 生长阶段的时间轴用去重后的时刻，避免重复点把时长算成 N 倍
    phase_t = np.unique(np.concatenate([np.arange(len(arr), dtype=float)
                                        for _, arr in valid]))
    phase_duration, _ = growth_phases(
        phase_t, fit["mu_fit"], fit["K"], fit["lambda_"], stable_fraction)

    result.update({
        "fit": fit,
        "merged_t": merged_t,
        "merged_counts": merged_counts,
        "phase_duration": phase_duration,
        "t90": time_at_fraction(stable_fraction, fit["mu_fit"], fit["K"], fit["lambda_"]),
    })
    return result
