import io
import sys
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from openpyxl.drawing.image import Image
from openpyxl.styles import numbers
from scipy.optimize import curve_fit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------- 全局字体配置 ----------------------
rcParams["font.family"] = ["Times New Roman", "serif"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 20
rcParams["xtick.labelsize"] = 20
rcParams["ytick.labelsize"] = 20
rcParams["axes.titlesize"] = 24
rcParams["legend.fontsize"] = 18
rcParams["axes.titley"] = 1.01


# ---------------------- 1. 定义修正Logistic生长模型 ----------------------
def modified_logistic_model(t, mu_fit, K, lambda_):
    """Return N_rel(t) after fitting the model on Y=ln(N_rel)."""
    t = np.asarray(t, dtype=float)
    if not np.isfinite(mu_fit) or not np.isfinite(K) or not np.isfinite(lambda_):
        return np.full_like(t, np.nan, dtype=float)

    amplitude = np.log(float(K)) if K > 1.0 else np.nan
    if not np.isfinite(amplitude) or amplitude <= 0:
        return np.full_like(t, np.nan, dtype=float)
    mu_fit = max(float(mu_fit), 1e-12)
    exponent = (4.0 * mu_fit / amplitude) * (float(lambda_) - t) + 2.0
    exponent = np.clip(exponent, -700, 700)
    y_fit = amplitude / (1.0 + np.exp(exponent))
    return np.exp(y_fit)


def _growth_curve_time_at_fraction(p, mu_fit, K, lambda_):
    if not np.isfinite(mu_fit) or not np.isfinite(K) or not np.isfinite(lambda_):
        return np.nan
    if p <= 0 or p >= 1:
        return np.nan

    amplitude = np.log(float(K)) if K > 1.0 else np.nan
    if not np.isfinite(amplitude) or amplitude <= 0 or mu_fit <= 0:
        return float(lambda_)

    return float(lambda_) + amplitude / (4.0 * float(mu_fit)) * (2.0 + np.log(p / (1.0 - p)))


def _specific_growth_rate_curve(t, mu_fit, K, lambda_):
    t = np.asarray(t, dtype=float)
    if not np.isfinite(mu_fit) or not np.isfinite(K) or not np.isfinite(lambda_):
        return np.full_like(t, np.nan, dtype=float)
    if K <= 1.0:
        return np.full_like(t, np.nan, dtype=float)

    amplitude = np.log(float(K))
    mu_fit = max(float(mu_fit), 1e-12)
    exponent = (4.0 * mu_fit / amplitude) * (float(lambda_) - t) + 2.0
    exponent = np.clip(exponent, -700, 700)
    exp_term = np.exp(exponent)
    return 4.0 * mu_fit * exp_term / (1.0 + exp_term) ** 2


def _true_specific_growth_rate_peak(mu_fit, K, lambda_):
    if not np.isfinite(mu_fit) or not np.isfinite(K) or not np.isfinite(lambda_):
        return np.nan, np.nan
    if K <= 1.0 or mu_fit <= 0:
        return np.nan, np.nan

    amplitude = np.log(float(K))
    t_peak = float(lambda_) + amplitude / (2.0 * float(mu_fit))
    return float(mu_fit), float(t_peak)


def calculate_growth_phases(t, mu_fit, K, lambda_, stable_fraction=0.90):
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return {
            "滞后期时长(h)": 0,
            "对数期时长(h)": 0,
            "稳定期时长(h)": 0,
        }, []

    t_start = float(t[0])
    t_end = float(t[-1])
    lag_end = float(lambda_) if np.isfinite(lambda_) else t_start
    stable_start = _growth_curve_time_at_fraction(stable_fraction, mu_fit, K, lambda_)
    if not np.isfinite(stable_start):
        stable_start = lag_end
    stable_start = max(stable_start, lag_end)

    lag_duration = max(min(lag_end, t_end) - t_start, 0.0)
    log_start = max(lag_end, t_start)
    log_end = min(stable_start, t_end)
    log_duration = max(log_end - log_start, 0.0)
    stable_duration = max(t_end - max(stable_start, t_start), 0.0)

    phase_duration = {
        "滞后期时长(h)": round(lag_duration, 2),
        "对数期时长(h)": round(log_duration, 2),
        "稳定期时长(h)": round(stable_duration, 2),
    }
    lag_phase_end = min(max(lag_end, t_start), t_end)
    stable_phase_start = min(max(stable_start, t_start), t_end)
    log_phase_start = lag_phase_end
    log_phase_end = min(max(stable_start, log_phase_start), t_end)
    phases = [
        {
            "阶段": "滞后期",
            "开始时间(h)": t_start,
            "结束时间(h)": lag_phase_end,
            "时长(h)": lag_duration,
        },
        {
            "阶段": "对数期",
            "开始时间(h)": log_phase_start,
            "结束时间(h)": log_phase_end,
            "时长(h)": log_duration,
        },
        {
            "阶段": "稳定期",
            "开始时间(h)": stable_phase_start,
            "结束时间(h)": t_end,
            "时长(h)": stable_duration,
        },
    ]
    return phase_duration, phases


# ---------------------- 2. 生成趋势线函数 ----------------------
def generate_trendline(x, y, degree=2):
    z = np.polyfit(x, y, degree)
    p = np.poly1d(z)
    return p(x)


def _empty_summary_row(sheet_name):
    return {
        "腔室名称": sheet_name,
        "最大比生长速率μmax (h^-1)": np.nan,
        "环境容纳量K (个/腔室)": np.nan,
        "初始细胞数量N0 (个/腔室)": np.nan,
        "滞止期参数λ(h)": np.nan,
        "拟合优度R²": np.nan,
        "平均细胞周期T_d(h)": np.nan,
        "增殖倍数 F": np.nan,
        "生长效率 η (1/增殖倍数)": np.nan,
        "滞后期时长(h)": np.nan,
        "对数期时长(h)": np.nan,
        "稳定期时长(h)": np.nan,
        "稳定期起始t90(h)": np.nan,
    }


def _finite_or_neg_inf(value):
    return value if value is not None and np.isfinite(value) else -np.inf


def _passes_merge_prefilter(summary_row):
    """Return whether an individual fit is usable for merged fitting."""
    individual_r2 = summary_row.get("拟合优度R²", np.nan)
    return (
        np.isfinite(individual_r2)
        and individual_r2 >= MIN_INDIVIDUAL_R2_FOR_MERGE
    )


def _to_float_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _safe_last_value(values, valid_num):
    if len(values) == 0:
        return np.nan
    index = min(max(valid_num - 1, 0), len(values) - 1)
    return values[index]


def _save_figure_to_buffer(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def _style_time_axis(ax):
    ax.set_xlabel("Cultivation time (h)")
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(3))
    y_tick_interval = (ax.get_yticks()[1] - ax.get_yticks()[0]) if len(ax.get_yticks()) > 1 else 1
    if y_tick_interval == 0:
        y_tick_interval = 1
    ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / 5))


def _insert_image(ws, buffer, anchor, width, height):
    img = Image(buffer)
    img.width = width
    img.height = height
    ws.add_image(img, anchor)


def _modified_logistic_fit_model(t, mu_fit, amplitude, lambda_):
    """Zwietering modified logistic model for Y(t)=ln(N_rel(t))."""
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


def _fit_logistic_curve(t, cell_counts, mu_guess=0.1, k_guess=None):
    t = np.asarray(t, dtype=float)
    cell_counts = np.asarray(cell_counts, dtype=float)

    if t.size < 3:
        raise ValueError("数据点不足，无法拟合")
    if not np.isfinite(t).all() or not np.isfinite(cell_counts).all():
        raise ValueError("数据中包含非数值")

    if np.any(cell_counts <= 0):
        raise ValueError("N_rel must be positive before taking log")

    max_count = float(np.max(cell_counts)) if cell_counts.size else 1.0
    if k_guess is None:
        k_guess = max(max_count, 1.0)
    mu_candidates = [mu_guess, max(mu_guess * 0.5, 1e-4), 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    last_error = None
    t_span = max(float(t[-1] - t[0]), 1.0)
    lambda_guess_index = min(len(t) - 1, max(1, int(round(len(t) * 0.2))))
    lambda_guess = float(t[lambda_guess_index]) if len(t) > 1 else float(t[0])
    k_upper = max(max_count * 10.0, float(k_guess) * 5.0, 1.000001)
    amplitude_guess = max(np.log(max(float(k_guess), 1.000001)), 1e-3)
    amplitude_upper = max(np.log(k_upper), 1.0)
    # Fit Y(t)=ln(N_rel(t)); its maximum slope is mu_max directly.
    bounds = (
        [1e-6, 1e-6, float(t[0]) - t_span],
        [50.0, amplitude_upper, float(t[-1]) + t_span],
    )

    for candidate_mu in mu_candidates:
        initial_guess = np.array([candidate_mu, amplitude_guess, lambda_guess], dtype=float)
        initial_guess = np.clip(initial_guess, np.array(bounds[0]) + 1e-12, np.array(bounds[1]) - 1e-12)
        try:
            popt, _ = curve_fit(
                f=_modified_logistic_fit_model,
                xdata=t,
                ydata=np.log(cell_counts),
                p0=initial_guess,
                bounds=bounds,
                maxfev=20000,
            )
            mu_fit, amplitude_fit, lambda_fit = popt
            K_fit = float(np.exp(amplitude_fit))
            y_fit = modified_logistic_model(t, mu_fit, K_fit, lambda_fit)
            fit_start_time = float(np.min(t))
            n0_fit = float(
                modified_logistic_model(
                    np.array([fit_start_time]), mu_fit, K_fit, lambda_fit
                )[0]
            )
            mu_max_true, mu_max_t = _true_specific_growth_rate_peak(mu_fit, K_fit, lambda_fit)
            ss_res = float(np.sum((cell_counts - y_fit) ** 2))
            ss_tot = float(np.sum((cell_counts - np.mean(cell_counts)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            t_d = (
                np.log(2) / mu_max_true
                if np.isfinite(mu_max_true) and mu_max_true > 0
                else np.nan
            )
            return {
                "mu_fit": float(mu_fit),
                "mu_max": float(mu_max_true),
                "mu_max_t": float(mu_max_t),
                "K": K_fit,
                "N0": n0_fit,
                "lambda_": float(lambda_fit),
                "r2": float(r2),
                "t_d": float(t_d),
                "y_fit": y_fit,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
            }
        except Exception as exc:
            last_error = exc

    raise last_error if last_error is not None else RuntimeError("拟合失败")


def _build_subset_fit(
    selected_sheets,
    record_map,
    original_order,
    valid_num,
    fit_cache,
):
    key = tuple(sheet for sheet in original_order if sheet in selected_sheets)
    if key in fit_cache:
        return fit_cache[key]

    selected_records = [record_map[sheet] for sheet in original_order if sheet in selected_sheets]
    if len(selected_records) < 2:
        fit_cache[key] = None
        return None

    merged_t = np.concatenate([record["t"][:valid_num] for record in selected_records])
    merged_counts = np.concatenate([record["cell_counts"][:valid_num] for record in selected_records])

    try:
        fit = _fit_logistic_curve(
            merged_t,
            merged_counts,
            mu_guess=0.1,
            k_guess=float(np.max(merged_counts)) if merged_counts.size else 1.0,
        )
    except Exception:
        fit_cache[key] = None
        return None

    sorted_t = np.sort(merged_t)
    phase_t = np.unique(np.concatenate([record["t"][:valid_num] for record in selected_records]))
    phase_duration, _ = calculate_growth_phases(phase_t, fit["mu_fit"], fit["K"], fit["lambda_"])
    t90 = _growth_curve_time_at_fraction(0.90, fit["mu_fit"], fit["K"], fit["lambda_"])
    residuals = {
        record["sheet"]: float(
            np.sum(
                (
                    record["cell_counts"][:valid_num]
                    - modified_logistic_model(record["t"][:valid_num], fit["mu_fit"], fit["K"], fit["lambda_"])
                )
                ** 2
            )
        )
        for record in selected_records
    }
    result = {
        "valid_sheets": [record["sheet"] for record in selected_records],
        "mu_fit": fit["mu_fit"],
        "mu_max": fit["mu_max"],
        "K": fit["K"],
        "N0": fit["N0"],
        "lambda_": fit["lambda_"],
        "r2": fit["r2"],
        "t_d": fit["t_d"],
        "t90": t90,
        "merged_t": merged_t,
        "merged_counts": merged_counts,
        "sorted_t": sorted_t,
        "phase_t": phase_t,
        "phase_duration": phase_duration,
        "residuals": residuals,
        "y_fit": fit["y_fit"],
    }
    fit_cache[key] = result
    return result


def _repair_subset(active_set, removed_order, record_map, original_order, valid_num, fit_cache, target_r2):
    current_result = _build_subset_fit(active_set, record_map, original_order, valid_num, fit_cache)
    if current_result is None:
        return None, active_set, removed_order

    while removed_order:
        candidates = []
        for sheet in reversed(removed_order):
            trial_set = set(active_set)
            trial_set.add(sheet)
            trial_result = _build_subset_fit(trial_set, record_map, original_order, valid_num, fit_cache)
            if trial_result and trial_result["r2"] >= target_r2:
                candidates.append((sheet, trial_result))

        if not candidates:
            break

        chosen_sheet, chosen_result = max(candidates, key=lambda item: item[1]["r2"])
        active_set = set(chosen_result["valid_sheets"])
        removed_order = [sheet for sheet in removed_order if sheet != chosen_sheet]
        current_result = chosen_result
        print(
            f"回看修复：重新加入腔室 {chosen_sheet}，当前R²={current_result['r2']:.6f}，"
            f"保留{len(active_set)}个腔室"
        )

    return current_result, active_set, removed_order


def _select_best_subset(record_map, original_order, valid_num, summary_map, target_r2=0.8):
    fit_cache = {}
    active_set = set(original_order)
    removed_order = []
    best_seen_result = None

    current_result = _build_subset_fit(active_set, record_map, original_order, valid_num, fit_cache)
    if current_result is not None:
        best_seen_result = current_result
        if current_result["r2"] >= target_r2:
            return current_result, best_seen_result

    while len(active_set) >= 2:
        active_order = [sheet for sheet in original_order if sheet in active_set]
        candidate_results = []
        for sheet in active_order:
            trial_set = set(active_set)
            trial_set.remove(sheet)
            trial_result = _build_subset_fit(trial_set, record_map, original_order, valid_num, fit_cache)
            candidate_results.append((sheet, trial_result))

        feasible_candidates = [
            (sheet, trial_result)
            for sheet, trial_result in candidate_results
            if trial_result is not None and trial_result["r2"] >= target_r2
        ]

        if feasible_candidates:
            chosen_sheet, chosen_result = max(feasible_candidates, key=lambda item: item[1]["r2"])
        else:
            chosen_sheet, chosen_result = max(
                candidate_results,
                key=lambda item: _finite_or_neg_inf(item[1]["r2"]) if item[1] is not None else -np.inf,
            )

        if chosen_result is None:
            if current_result is not None and current_result["residuals"]:
                chosen_sheet = max(current_result["residuals"], key=current_result["residuals"].get)
            else:
                chosen_sheet = min(
                    active_order,
                    key=lambda sheet: _finite_or_neg_inf(summary_map.get(sheet, {}).get("拟合优度R²")),
                )
            trial_set = set(active_set)
            trial_set.remove(chosen_sheet)
            chosen_result = _build_subset_fit(trial_set, record_map, original_order, valid_num, fit_cache)

        active_set.remove(chosen_sheet)
        removed_order.append(chosen_sheet)
        current_result = chosen_result
        if current_result is not None:
            if (
                best_seen_result is None
                or _finite_or_neg_inf(current_result["r2"]) > _finite_or_neg_inf(best_seen_result["r2"])
            ):
                best_seen_result = current_result
            print(
                f"后向剔除：移除腔室 {chosen_sheet}，重拟合后R²={current_result['r2']:.6f}，"
                f"剩余{len(active_set)}个腔室"
            )
        else:
            print(f"后向剔除：移除腔室 {chosen_sheet} 后拟合失败，剩余{len(active_set)}个腔室")

        if current_result is not None and current_result["r2"] >= target_r2:
            repaired_result, repaired_set, repaired_removed = _repair_subset(
                active_set,
                removed_order,
                record_map,
                original_order,
                valid_num,
                fit_cache,
                target_r2,
            )
            return repaired_result, best_seen_result

    if current_result is not None and current_result["r2"] >= target_r2:
        repaired_result, _, _ = _repair_subset(
            active_set,
            removed_order,
            record_map,
            original_order,
            valid_num,
            fit_cache,
            target_r2,
        )
        return repaired_result, best_seen_result

    return None, best_seen_result


def _fit_and_write_individual_sheet(
    ws,
    sheet,
    plt_name,
    t,
    cell_counts,
    valid_num,
    df,
):
    fit_result = _fit_logistic_curve(t, cell_counts)
    phase_duration, _ = calculate_growth_phases(t, fit_result["mu_fit"], fit_result["K"], fit_result["lambda_"])
    t90 = _growth_curve_time_at_fraction(0.90, fit_result["mu_fit"], fit_result["K"], fit_result["lambda_"])
    lag_duration = phase_duration["滞后期时长(h)"]
    log_duration = phase_duration["对数期时长(h)"]
    stable_duration = phase_duration["稳定期时长(h)"]
    F_last_cell_number = _safe_last_value(cell_counts, valid_num)
    growth_rate = (
        F_last_cell_number / valid_num
        if F_last_cell_number is not None and np.isfinite(F_last_cell_number)
        else np.nan
    )

    row = {
        "腔室名称": sheet,
        "最大比生长速率μmax (h^-1)": round(fit_result["mu_max"], 4),
        "环境容纳量K (个/腔室)": round(fit_result["K"], 2),
        "初始细胞数量N0 (个/腔室)": fit_result["N0"],
        "滞止期参数λ(h)": round(fit_result["lambda_"], 4),
        "拟合优度R²": round(fit_result["r2"], 4),
        "平均细胞周期T_d(h)": round(fit_result["t_d"], 2) if np.isfinite(fit_result["t_d"]) else np.nan,
        "增殖倍数 F": round(F_last_cell_number, 2) if np.isfinite(F_last_cell_number) else np.nan,
        "生长效率 η (1/增殖倍数)": round(growth_rate, 3) if np.isfinite(growth_rate) else np.nan,
        "滞后期时长(h)": lag_duration,
        "对数期时长(h)": log_duration,
        "稳定期时长(h)": stable_duration,
        "稳定期起始t90(h)": round(t90, 2) if np.isfinite(t90) else np.nan,
    }

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, cell_counts, label="Actual data", color="blue", alpha=0.6)
    plt.plot(t, fit_result["y_fit"], label="Fitted curve", color="red", linewidth=2)
    ax = plt.gca()
    ax.set_xlabel("Cultivation time (h)")
    ax.set_ylabel("Relative cell number (normalized to 0 h)")
    ax.set_title(f"{plt_name} Cell growth curve fitting")
    _style_time_axis(ax)
    param_text = (
        f"$\\it{{μ}}_{{\\mathrm{{max}}}}$= {round(fit_result['mu_max'], 4)} h⁻¹\n"
        f"$\\it{{K}}$ = {round(fit_result['K'], 2)}\n"
        f"$\\it{{N}}_{{0}}$ = {round(fit_result['N0'], 2)}\n"
        f"$\\it{{λ}}$ = {round(fit_result['lambda_'], 2)}\n"
        f"R² = {round(fit_result['r2'], 4)}"
    )
    plt.text(
        0.05,
        0.95,
        param_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend()
    plt.grid(alpha=0.3)
    img_buffer = _save_figure_to_buffer(fig)

    param_col_start = 5
    ws.cell(row=1, column=param_col_start).value = "参数名"
    ws.cell(row=1, column=param_col_start + 1).value = "参数值"
    ws.cell(row=2, column=param_col_start).value = "最大比生长速率uMAX(h^-1)"
    ws.cell(row=2, column=param_col_start + 1).value = round(fit_result["mu_max"], 4)
    ws.cell(row=3, column=param_col_start).value = "环境容纳量K(个)"
    ws.cell(row=3, column=param_col_start + 1).value = round(fit_result["K"], 4)
    ws.cell(row=4, column=param_col_start).value = "初始细胞数量N0(个)"
    ws.cell(row=4, column=param_col_start + 1).value = round(fit_result["N0"], 4)
    ws.cell(row=5, column=param_col_start).value = "滞止期参数λ(h)"
    ws.cell(row=5, column=param_col_start + 1).value = round(fit_result["lambda_"], 4)
    ws.cell(row=6, column=param_col_start).value = "拟合优度R²"
    ws.cell(row=6, column=param_col_start + 1).value = round(fit_result["r2"], 4)
    ws.cell(row=7, column=param_col_start).value = "平均细胞周期T_d(h)"
    ws.cell(row=7, column=param_col_start + 1).value = round(fit_result["t_d"], 2) if np.isfinite(fit_result["t_d"]) else np.nan
    ws.cell(row=8, column=param_col_start).value = "增殖倍数 F"
    ws.cell(row=8, column=param_col_start + 1).value = round(F_last_cell_number, 2) if np.isfinite(F_last_cell_number) else np.nan
    ws.cell(row=9, column=param_col_start).value = "生长效率 η (1/增殖倍数)"
    ws.cell(row=9, column=param_col_start + 1).value = round(growth_rate, 3) if np.isfinite(growth_rate) else np.nan
    ws.cell(row=10, column=param_col_start).value = "滞后期时长(h)"
    ws.cell(row=10, column=param_col_start + 1).value = lag_duration
    ws.cell(row=11, column=param_col_start).value = "对数期时长(h)"
    ws.cell(row=11, column=param_col_start + 1).value = log_duration
    ws.cell(row=12, column=param_col_start).value = "稳定期时长(h)"
    ws.cell(row=12, column=param_col_start + 1).value = stable_duration
    ws.cell(row=13, column=param_col_start).value = "稳定期起始t90(h)"
    ws.cell(row=13, column=param_col_start + 1).value = round(t90, 2) if np.isfinite(t90) else np.nan
    _insert_image(ws, img_buffer, "H02", 600, 400)

    return row


def _result_priority(result):
    if result is None:
        return (-np.inf, -np.inf, -np.inf, -np.inf)
    residuals = result.get("residuals") or {}
    r2 = _finite_or_neg_inf(result.get("r2"))
    ss_res = result.get("ss_res", np.inf)
    total_residual = -float(ss_res) if np.isfinite(ss_res) else -np.inf
    max_residual = -float(max(residuals.values())) if residuals else -np.inf
    count = len(result.get("valid_sheets", []))
    # Among feasible results, retain as many chambers as possible before
    # comparing fit quality.
    return (count, r2, total_residual, max_residual)


def _state_priority(state):
    return _result_priority(state.get("result"))


def _quality_priority(result):
    """Rank an infeasible result by fit quality for diagnostics."""
    if result is None:
        return (-np.inf, -np.inf, -np.inf)
    r2 = _finite_or_neg_inf(result.get("r2"))
    count = len(result.get("valid_sheets", []))
    ss_res = result.get("ss_res", np.inf)
    return (r2, count, -float(ss_res) if np.isfinite(ss_res) else -np.inf)


def _removal_score(sheet, result, record_map, summary_map, target_r2):
    """Estimate how likely a chamber is to harm the merged fit."""
    residual = float((result.get("residuals") or {}).get(sheet, 0.0))
    counts = np.asarray(record_map[sheet]["cell_counts"], dtype=float)
    counts = counts[np.isfinite(counts)]
    signal_scale = float(np.sum((counts - 1.0) ** 2)) if counts.size else 0.0
    normalized_residual = residual / max(signal_scale, 1e-12)

    individual_r2 = _finite_or_neg_inf(
        summary_map.get(sheet, {}).get("拟合优度R²")
    )
    quality_penalty = (
        max(target_r2 - individual_r2, 0.0) / max(target_r2, 1e-12)
        if np.isfinite(individual_r2)
        else 1.0
    )

    merged_mu = result.get("mu_max", np.nan)
    merged_k = result.get("K", np.nan)
    chamber_mu = _finite_or_neg_inf(
        summary_map.get(sheet, {}).get("最大比生长速率μmax (h^-1)")
    )
    chamber_k = _finite_or_neg_inf(
        summary_map.get(sheet, {}).get("环境容纳量K (个/腔室)")
    )
    parameter_penalty = 0.0
    if (
        np.isfinite(merged_mu)
        and merged_mu > 0
        and np.isfinite(chamber_mu)
        and chamber_mu > 0
    ):
        parameter_penalty += abs(np.log(chamber_mu / merged_mu))
    if (
        np.isfinite(merged_k)
        and merged_k > 1
        and np.isfinite(chamber_k)
        and chamber_k > 1
    ):
        parameter_penalty += abs(np.log(chamber_k / merged_k))

    return (
        0.70 * np.log1p(max(normalized_residual, 0.0))
        + 0.20 * quality_penalty
        + 0.10 * parameter_penalty
    )


def _rank_removal_candidates(state, active_order, summary_map, branch_limit):
    branch_limit = max(1, min(branch_limit, len(active_order)))
    result = state.get("result")
    if result is not None and result.get("residuals"):
        ordered = sorted(
            active_order,
            key=lambda sheet: (
                _finite_or_neg_inf(result["residuals"].get(sheet)),
                -_finite_or_neg_inf(summary_map.get(sheet, {}).get("拟合优度R²")),
                sheet,
            ),
            reverse=True,
        )
    else:
        ordered = sorted(
            active_order,
            key=lambda sheet: (
                _finite_or_neg_inf(summary_map.get(sheet, {}).get("拟合优度R²")),
                sheet,
            ),
        )
    return ordered[:branch_limit]


def _exchange_repair(
    current_result,
    record_map,
    original_order,
    valid_num,
    fit_cache,
    target_r2,
    max_rounds,
):
    """Try a small number of one-for-one swaps after greedy selection."""
    if current_result is None or current_result["r2"] < target_r2:
        return current_result

    active_set = set(current_result["valid_sheets"])
    for round_index in range(max_rounds):
        removed_set = set(original_order) - active_set
        candidates = []
        for outgoing in active_set:
            for incoming in removed_set:
                trial_set = (active_set - {outgoing}) | {incoming}
                trial_result = _build_subset_fit(
                    trial_set,
                    record_map,
                    original_order,
                    valid_num,
                    fit_cache,
                )
                if trial_result is not None and trial_result["r2"] >= target_r2:
                    candidates.append(trial_result)

        if not candidates:
            break

        current_result = max(candidates, key=_result_priority)
        active_set = set(current_result["valid_sheets"])
        print(
            f"Exchange repair round {round_index + 1}: "
            f"R²={current_result['r2']:.6f}"
        )

    return current_result


def _select_best_subset_beam(
    record_map,
    original_order,
    valid_num,
    summary_map,
    target_r2=0.8,
    beam_width=None,
    branch_limit=None,
):
    """Fast greedy selection with bounded repair.

    The method prioritizes chamber count while avoiding exponential subset
    enumeration. It is a practical approximation to the global maximum-count
    solution: greedy removal finds a feasible subset, add-back restores safe
    chambers, and a few one-for-one swaps repair local mistakes.
    """
    fit_cache = {}
    all_order = list(original_order)
    if len(all_order) < 2:
        return None, None

    trial_limit = branch_limit or MERGE_REMOVAL_TRIALS
    trial_limit = max(1, int(trial_limit))
    fit_count = 0
    current_result = _build_subset_fit(
        set(all_order), record_map, all_order, valid_num, fit_cache
    )
    fit_count += 1
    best_seen_result = current_result

    if current_result is not None and current_result["r2"] >= target_r2:
        print(
            f"Fast subset search: evaluated {fit_count} fits; "
            f"selected {len(current_result['valid_sheets'])} chambers, "
            f"R²={current_result['r2']:.6f}"
        )
        return current_result, best_seen_result

    active_set = set(all_order)
    removed_order = []
    while len(active_set) > 2 and current_result is not None:
        active_order = [sheet for sheet in all_order if sheet in active_set]
        ranked = sorted(
            active_order,
            key=lambda sheet: _removal_score(
                sheet, current_result, record_map, summary_map, target_r2
            ),
            reverse=True,
        )
        candidate_sheets = ranked[: min(trial_limit, len(ranked))]
        trial_candidates = []
        for sheet in candidate_sheets:
            trial_result = _build_subset_fit(
                active_set - {sheet},
                record_map,
                all_order,
                valid_num,
                fit_cache,
            )
            fit_count += 1
            if trial_result is not None:
                trial_candidates.append((sheet, trial_result))
                if _quality_priority(trial_result) > _quality_priority(best_seen_result):
                    best_seen_result = trial_result

        if not trial_candidates:
            break

        removed_sheet, current_result = max(
            trial_candidates,
            key=lambda item: _quality_priority(item[1]),
        )
        active_set = set(current_result["valid_sheets"])
        removed_order.append(removed_sheet)
        print(
            f"Greedy removal: removed {removed_sheet}; "
            f"retaining {len(active_set)} chambers, "
            f"R²={current_result['r2']:.6f}, fits={fit_count}"
        )

        if current_result["r2"] >= target_r2:
            break

    if current_result is None or current_result["r2"] < target_r2:
        return None, best_seen_result

    repaired_result, _, _ = _repair_subset(
        active_set,
        removed_order,
        record_map,
        all_order,
        valid_num,
        fit_cache,
        target_r2,
    )
    current_result = repaired_result or current_result
    current_result = _exchange_repair(
        current_result,
        record_map,
        all_order,
        valid_num,
        fit_cache,
        target_r2,
        MERGE_MAX_EXCHANGE_ROUNDS,
    )

    repaired_result, _, _ = _repair_subset(
        set(current_result["valid_sheets"]),
        [sheet for sheet in all_order if sheet not in current_result["valid_sheets"]],
        record_map,
        all_order,
        valid_num,
        fit_cache,
        target_r2,
    )
    final_result = repaired_result or current_result
    print(
        f"Fast subset search complete: evaluated {len(fit_cache)} unique fits; "
        f"selected {len(final_result['valid_sheets'])} chambers, "
        f"R²={final_result['r2']:.6f}"
    )
    return final_result, best_seen_result


def process_cell_growth(
    excel_path,
    result_excel_path="enhanced_cell_growth_results.xlsx",
    min_data_points=5,
    experimental_parameters="test",
    skip_sheet_name={"数据汇总"},
    valid_num=96,
):
    skip_sheet = skip_sheet_name
    wb_original = openpyxl.load_workbook(excel_path)
    excel_file = pd.ExcelFile(excel_path)

    try:
        sheet_names = wb_original.sheetnames
        individual_summary = []
        individual_data = []
        record_map = {}
        summary_map = {}
        eligible_order = []
        merged_params = None
        merged_img_buffer = None
        target_r2 = 0.8

        for sheet in sheet_names:
            if sheet in skip_sheet:
                print(f"跳过页签: {sheet}")
                continue

            print(f"正在处理页签: {sheet}")
            plt_name = experimental_parameters + "_" + sheet[4:]
            ws = wb_original[sheet]
            try:
                df = excel_file.parse(sheet_name=sheet, nrows=valid_num + 1)
            except Exception as e:
                print(f"页签 {sheet} 读取失败：{str(e)}\n")
                row = _empty_summary_row(sheet)
                individual_summary.append(row)
                summary_map[sheet] = row
                continue

            required_columns = ["目标数量", "总面积(μm²)", "相对平均细胞面积"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"页签 {sheet} 缺少必要列: {missing_cols}，跳过\n")
                row = _empty_summary_row(sheet)
                individual_summary.append(row)
                summary_map[sheet] = row
                continue

            cell_counts = _to_float_array(df["目标数量"])
            if cell_counts.size == 0 or np.all(np.nan_to_num(cell_counts, nan=0.0) == 0):
                print(f"页签 {sheet} 所有细胞数量为0，跳过\n")
                row = _empty_summary_row(sheet)
                individual_summary.append(row)
                summary_map[sheet] = row
                continue

            t = np.arange(len(cell_counts), dtype=float)
            record = {
                "sheet": sheet,
                "t": t,
                "cell_counts": cell_counts,
                "area": _to_float_array(df["总面积(μm²)"]),
                "avg_cell_area": _to_float_array(df["相对平均细胞面积"]),
            }

            if len(cell_counts) >= min_data_points:
                individual_data.append(record)

            try:
                row = _fit_and_write_individual_sheet(
                    ws=ws,
                    sheet=sheet,
                    plt_name=plt_name,
                    t=t,
                    cell_counts=cell_counts,
                    valid_num=valid_num,
                    df=df,
                )
                individual_summary.append(row)
                summary_map[sheet] = row

                if len(cell_counts) >= min_data_points:
                    individual_r2 = row.get("拟合优度R²", np.nan)
                    if _passes_merge_prefilter(row):
                        eligible_order.append(sheet)
                        record_map[sheet] = record
                        print(f"页签 {sheet} 单独拟合完成，进入汇总候选集\n")
                    else:
                        print(
                            f"页签 {sheet} 单独拟合完成，但R²={individual_r2} "
                            f"< {MIN_INDIVIDUAL_R2_FOR_MERGE}，不参与汇总拟合\n"
                        )
                else:
                    print(f"页签 {sheet} 单独拟合完成，但数据点不足，不参与汇总拟合\n")
            except Exception as e:
                print(f"页签 {sheet} 单独拟合失败：{str(e)}\n")
                row = _empty_summary_row(sheet)
                individual_summary.append(row)
                summary_map[sheet] = row

        if len(eligible_order) >= 2:
            merged_params, best_merged_result = _select_best_subset_beam(
                record_map=record_map,
                original_order=eligible_order,
                valid_num=valid_num,
                summary_map=summary_map,
                target_r2=target_r2,
            )
            if merged_params is not None:
                merged_img_buffer = None
                print(
                    f"合并拟合完成（已预筛选并按最大腔室数搜索）："
                    f"μmax={merged_params['mu_max']:.6f}, "
                    f"K={merged_params['K']:.6f}, "
                    f"N0={merged_params['N0']:.6f}, "
                    f"λ={merged_params['lambda_']:.6f}, "
                    f"R²={merged_params['r2']:.6f}"
                )

                fig = plt.figure(figsize=(12, 7))
                plt.scatter(
                    merged_params["merged_t"],
                    merged_params["merged_counts"],
                    label="Merged actual data",
                    color="blue",
                    alpha=0.5,
                    s=30,
                )
                sorted_t = merged_params["sorted_t"]
                plt.plot(
                    sorted_t,
                    modified_logistic_model(
                        sorted_t,
                        merged_params["mu_fit"],
                        merged_params["K"],
                        merged_params["lambda_"],
                    ),
                    label="Merged fitted curve",
                    color="#F2BA02",
                    linewidth=2,
                )
                ax = plt.gca()
                ax.set_xlabel("Cultivation time (h)")
                ax.set_ylabel("Relative cell number (normalized to 0 h)")
                ax.set_title(
                    f"Growth curve (merged data, {len(merged_params['valid_sheets'])} chambers, "
                    # f"NH$_4^+$-N={experimental_parameters})"
                    f"{experimental_parameters})"
                )
                _style_time_axis(ax)
                param_text = (
                    f"$\\it{{μ}}_{{\\mathrm{{max}}}}$ = {merged_params['mu_max']:.4f} h⁻¹\n"
                    f"$\\it{{K}}$ = {merged_params['K']:.2f}\n"
                    f"$\\it{{N}}_{{0}}$ = {merged_params['N0']:.2f}\n"
                    f"$\\it{{λ}}$ = {merged_params['lambda_']:.2f}\n"
                    f"R² = {merged_params['r2']:.4f}"
                )
                plt.text(
                    0.05,
                    0.95,
                    param_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=20,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                plt.legend()
                plt.grid(alpha=0.3)
                merged_img_buffer = _save_figure_to_buffer(fig)
                merged_params["phase_duration"], _ = calculate_growth_phases(
                    merged_params["phase_t"],
                    merged_params["mu_fit"],
                    merged_params["K"],
                    merged_params["lambda_"],
                )
            else:
                if best_merged_result is not None:
                    print(
                        f"无法找到满足R²≥{target_r2}的合并腔室组合，"
                        f"最佳结果为R²={best_merged_result['r2']:.2f}，"
                        f"保留{len(best_merged_result['valid_sheets'])}个腔室"
                    )
                else:
                    print("无法得到可用的合并拟合结果")

        if "汇总结果" in wb_original.sheetnames:
            del wb_original["汇总结果"]
        summary_ws = wb_original.create_sheet(title="汇总结果", index=0)
        summary_ws["A1"] = "腔室名称"
        summary_ws["B1"] = "最大比生长速率μmax (h^-1)"
        summary_ws["C1"] = "环境容纳量K (个/腔室)"
        summary_ws["D1"] = "初始细胞数量N0 (个/腔室)"
        summary_ws["E1"] = "拟合优度R²"
        summary_ws["F1"] = "平均细胞周期T_d(h)"
        summary_ws["G1"] = "增殖倍数 F"
        summary_ws["H1"] = "生长效率 η (1/增殖倍数)"
        summary_ws["I1"] = "滞后期时长(h)"
        summary_ws["J1"] = "对数期时长(h)"
        summary_ws["K1"] = "稳定期时长(h)"
        summary_ws["L1"] = "滞止期参数λ(h)"

        selected_sheets = merged_params.get("valid_sheets", []) if merged_params else []
        valid_data = [item for item in individual_summary if item["腔室名称"] in selected_sheets]
        non_valid_data = [item for item in individual_summary if item["腔室名称"] not in selected_sheets]

        current_row = 2
        for data in valid_data:
            summary_ws[f"A{current_row}"] = data["腔室名称"]
            summary_ws[f"B{current_row}"] = data["最大比生长速率μmax (h^-1)"]
            summary_ws[f"C{current_row}"] = data["环境容纳量K (个/腔室)"]
            summary_ws[f"D{current_row}"] = data["初始细胞数量N0 (个/腔室)"]
            summary_ws[f"E{current_row}"] = data["拟合优度R²"]
            summary_ws[f"F{current_row}"] = data["平均细胞周期T_d(h)"]
            summary_ws[f"G{current_row}"] = data["增殖倍数 F"]
            summary_ws[f"H{current_row}"] = data["生长效率 η (1/增殖倍数)"]
            summary_ws[f"I{current_row}"] = data["滞后期时长(h)"]
            summary_ws[f"J{current_row}"] = data["对数期时长(h)"]
            summary_ws[f"K{current_row}"] = data["稳定期时长(h)"]
            summary_ws[f"L{current_row}"] = data["滞止期参数λ(h)"]
            current_row += 1

        if non_valid_data:
            summary_ws[f"A{current_row}"] = "未参与拟合数据"
            summary_ws.merge_cells(f"A{current_row}:L{current_row}")
            summary_ws[f"A{current_row}"].font = openpyxl.styles.Font(bold=True, color="FF0000")
            current_row += 1

        for data in non_valid_data:
            summary_ws[f"A{current_row}"] = data["腔室名称"]
            summary_ws[f"B{current_row}"] = data["最大比生长速率μmax (h^-1)"]
            summary_ws[f"C{current_row}"] = data["环境容纳量K (个/腔室)"]
            summary_ws[f"D{current_row}"] = data["初始细胞数量N0 (个/腔室)"]
            summary_ws[f"E{current_row}"] = data["拟合优度R²"]
            summary_ws[f"F{current_row}"] = data["平均细胞周期T_d(h)"]
            summary_ws[f"G{current_row}"] = data["增殖倍数 F"]
            summary_ws[f"H{current_row}"] = data["生长效率 η (1/增殖倍数)"]
            summary_ws[f"I{current_row}"] = data["滞后期时长(h)"]
            summary_ws[f"J{current_row}"] = data["对数期时长(h)"]
            summary_ws[f"K{current_row}"] = data["稳定期时长(h)"]
            summary_ws[f"L{current_row}"] = data["滞止期参数λ(h)"]
            current_row += 1

        for col in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            summary_ws.column_dimensions[col].width = 25
        for col in ["I", "J", "K", "L"]:
            summary_ws.column_dimensions[col].width = 20

        reserve_row = current_row + 1

        if merged_params is not None:
            last_row = current_row + 1
            summary_ws[f"A{last_row}"] = "合并拟合结果"
            summary_ws[f"A{last_row}"].font = openpyxl.styles.Font(bold=True)
            merged_phase_duration = merged_params["phase_duration"]
            param_rows = {
                "参与拟合的页签数量": len(merged_params["valid_sheets"]),
                "总数据点数量": len(merged_params["merged_t"]),
                "最大比生长速率μmax (h^-1)": round(merged_params["mu_max"], 4),
                "环境容纳量K (个/腔室)": round(merged_params["K"], 2),
                "初始细胞数量N0 (个/腔室)": round(merged_params["N0"], 2),
                "滞止期参数λ(h)": round(merged_params["lambda_"], 2),
                "拟合优度R²": round(merged_params["r2"], 4),
                "滞后期时长(h)": merged_phase_duration["滞后期时长(h)"],
                "对数期时长(h)": merged_phase_duration["对数期时长(h)"],
                "稳定期时长(h)": merged_phase_duration["稳定期时长(h)"],
                "稳定期起始t90(h)": round(merged_params["t90"], 2) if np.isfinite(merged_params.get("t90", np.nan)) else np.nan,
            }
            current_row = last_row + 1
            for param, value in param_rows.items():
                summary_ws[f"A{current_row}"] = param
                summary_ws[f"B{current_row}"] = value
                current_row += 1
            if merged_img_buffer is not None:
                summary_ws[f"C{last_row}"] = "参与拟合的腔室"
                summary_ws[f"D{last_row}"] = ", ".join(merged_params["valid_sheets"])
                _insert_image(summary_ws, merged_img_buffer, f"C{last_row + 1}", 700, 500)

        if merged_params is not None and len(selected_sheets) > 0:
            for sheet_data in individual_data:
                sheet_name = sheet_data["sheet"]
                plt_name = experimental_parameters + "_" + sheet_name[4:]
                t_data = sheet_data["t"]
                counts_data = sheet_data["cell_counts"]
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(t_data, counts_data, label=f"Actual data", color="blue", alpha=0.6)
                merged_curve = modified_logistic_model(
                    t_data,
                    merged_params["mu_fit"],
                    merged_params["K"],
                    merged_params["lambda_"],
                )
                plt.plot(t_data, merged_curve, label="Merged fitted curve", color="#F2BA02", linewidth=2)
                ax = plt.gca()
                ax.set_xlabel("Cultivation time (h)")
                ax.set_ylabel("Relative cell number (normalized to 0 h)")
                ax.set_title(f"{plt_name} data vs merged fitting curve")
                _style_time_axis(ax)
                param_text = (
                    f"$\\it{{μ}}_{{\\mathrm{{max}}}}$ = {merged_params['mu_max']:.4f} h⁻¹\n"
                    f"$\\it{{K}}$ = {merged_params['K']:.2f}\n"
                    f"$\\it{{N}}_{{0}}$ = {merged_params['N0']:.2f}\n"
                    f"$\\it{{λ}}$ = {merged_params['lambda_']:.2f}\n"
                    f"R² = {merged_params['r2']:.4f}"
                )
                plt.text(
                    0.05,
                    0.95,
                    param_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=20,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                plt.legend()
                plt.grid(alpha=0.3)
                img_buffer = _save_figure_to_buffer(fig)
                try:
                    ws = wb_original[sheet_name]
                    _insert_image(ws, img_buffer, "H24", 600, 400)
                    print(f"已为 {sheet_name} 添加合并拟合对比图")
                except Exception as e:
                    print(f"为 {sheet_name} 添加对比图失败：{str(e)}")

        for sheet_data in individual_data:
            sheet_name = sheet_data["sheet"]
            t_data = sheet_data["t"][:valid_num]
            area = sheet_data["area"][:valid_num]
            avg_cell_area = sheet_data["avg_cell_area"][:valid_num]
            try:
                ws = wb_original[sheet_name]
                plt_title = experimental_parameters + "_" + sheet_name[4:]

                fig = plt.figure(figsize=(10, 6))
                plt.scatter(t_data, area, label="Relative total cell area", color="green", alpha=0.6)
                ax = plt.gca()
                ax.set_xlabel("Cultivation time (h)")
                ax.set_ylabel("Relative total cell area (normalized to 0 h)")
                ax.set_title(f"{plt_title} - Relative total cell area variation")
                _style_time_axis(ax)
                plt.legend()
                plt.grid(alpha=0.3)
                area_buffer = _save_figure_to_buffer(fig)
                _insert_image(ws, area_buffer, "R2", 600, 400)

                fig = plt.figure(figsize=(10, 6))
                plt.scatter(t_data, avg_cell_area, label="Relative average cell area", color="purple", alpha=0.6, s=30)
                plt.plot(t_data, avg_cell_area, color="purple", alpha=0.8, linestyle="-", linewidth=1.5, marker="", markersize=5)
                ax = plt.gca()
                ax.set_xlabel("Cultivation time (h)")
                ax.set_ylabel("Relative average cell area (normalized to 0 h)")
                ax.set_title(f"{plt_title} - Relative average cell area variation")
                _style_time_axis(ax)
                plt.legend()
                plt.grid(alpha=0.3)
                avg_buffer = _save_figure_to_buffer(fig)
                _insert_image(ws, avg_buffer, "R24", 600, 400)
                print(f"已为 {sheet_name} 添加2个图表")
            except Exception as e:
                print(f"为 {sheet_name} 添加趋势图表失败：{str(e)}")

        if merged_params is not None and len(selected_sheets) > 0:
            all_t = []
            all_area = []
            all_avg_area = []
            for sheet_data in individual_data:
                if sheet_data["sheet"] in merged_params["valid_sheets"]:
                    all_t.extend(sheet_data["t"][:valid_num])
                    all_area.extend(sheet_data["area"][:valid_num])
                    all_avg_area.extend(sheet_data["avg_cell_area"][:valid_num])

            fig = plt.figure(figsize=(12, 7))
            plt.scatter(all_t, all_area, label="Relative total area data", color="green", alpha=0.6, s=30)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative total cell area (normalized to 0 h)")
            ax.set_title(
                f"Relative total area (merged data, {len(merged_params['valid_sheets'])} chambers, "
                # f"NH$_4^+$-N={experimental_parameters})"
                f"{experimental_parameters})"
            )
            _style_time_axis(ax)
            plt.legend()
            plt.grid(alpha=0.3)
            total_area_buffer = _save_figure_to_buffer(fig)

            fig = plt.figure(figsize=(12, 7))
            plt.scatter(all_t, all_avg_area, label="Relative average cell area data", color="purple", alpha=0.6, s=30)
            ax = plt.gca()
            ax.set_xlabel("Cultivation time (h)")
            ax.set_ylabel("Relative average cell area (normalized to 0 h)")
            ax.set_title(
                f"Relative average cell area (merged data, {len(merged_params['valid_sheets'])} chambers, "
                # f"NH$_4^+$-N={experimental_parameters})"
                f"{experimental_parameters})"
            )
            _style_time_axis(ax)
            plt.legend()
            plt.grid(alpha=0.3)
            avg_area_buffer = _save_figure_to_buffer(fig)

            current_row = reserve_row + 1
            _insert_image(summary_ws, total_area_buffer, f"G{current_row}", 700, 500)
            _insert_image(summary_ws, avg_area_buffer, f"L{current_row}", 700, 500)
            print("已在汇总结果页签添加2个汇总散点图")

        for sheet_data in individual_data:
            sheet_name = sheet_data["sheet"]
            try:
                ws = wb_original[sheet_name]
                title_row = 1
                area_col = None
                avg_area_col = None
                for col in range(1, ws.max_column + 1):
                    cell_value = ws.cell(row=title_row, column=col).value
                    if cell_value == "总面积(μm²)":
                        area_col = col
                    elif cell_value == "相对平均细胞面积":
                        avg_area_col = col
                    if area_col and avg_area_col:
                        break

                if area_col:
                    for row in range(2, ws.max_row + 1):
                        cell = ws.cell(row=row, column=area_col)
                        if cell.value is not None and isinstance(cell.value, (int, float)):
                            cell.value = round(cell.value, 2)
                            cell.number_format = numbers.FORMAT_NUMBER_00

                if avg_area_col:
                    for row in range(2, ws.max_row + 1):
                        cell = ws.cell(row=row, column=avg_area_col)
                        if cell.value is not None and isinstance(cell.value, (int, float)):
                            cell.value = round(cell.value, 2)
                            cell.number_format = numbers.FORMAT_NUMBER_00

                print(f"已为 {sheet_name} 格式化面积数据为两位小数")
            except Exception as e:
                print(f"格式化 {sheet_name} 面积数据时出错：{str(e)}")

        wb_original.save(result_excel_path)
        print(f"所有处理完成，结果已保存到：{result_excel_path}")
        return pd.DataFrame(individual_summary)
    finally:
        try:
            excel_file.close()
        except Exception:
            pass
        wb_original.close()

"""
该代码为（202600822）汇总拟合流程：
现在汇总拟合流程为：
1. 先拟合全部候选腔室；
2. 若 R² >= 0.8，直接保留全部腔室；
3. 若不达标，根据以下指标给腔室评分：
   - 对整体拟合残差的贡献；
   - 单腔室 R²；
   - 单腔室 μmax、K 与整体参数的偏离程度；
4. 每轮只测试评分最高的前 3 个可疑腔室；
5. 选择能使 R² 改善最多的删除方案；
6. 达到 R² >= 0.8 后，逐个尝试加回已删除腔室；
7. 最后进行最多 2 轮一对一交换修复。
"""
# ================================== 【配置区：所有可修改参数都在这里】 ==================================
DATE_STR = "20260712"
VALID_HOURS = "96"
# 1. 原始标准化Excel所在的输入文件夹（路径中的日期自动引用上面的变量）
INPUT_FOLDER = rf"F:\Microalgae_Photoes\0828_μmax_revision\{DATE_STR}\数据汇总\02_标准化数据"
# 2. 可视化结果Excel的保存文件夹
OUTPUT_FOLDER = rf"F:\Microalgae_Photoes\0828_μmax_revision\{DATE_STR}\数据汇总\03_可视化结果\{VALID_HOURS}小时"
# 3. 所有Excel共用的全局参数（和原代码参数含义完全一致）
SKIP_SHEET = {"数据汇总"}              # 需要跳过的工作表名称
SET_VALID_NUM = int(VALID_HOURS) + 1 # 有效数量阈值
MIN_DATA_POINTS = 5               # 最少数据点数
# Only clearly poor individual fits are excluded before merged search.
MIN_INDIVIDUAL_R2_FOR_MERGE = 0.20
# Number of suspicious chambers tested at each greedy-removal round.
MERGE_REMOVAL_TRIALS = 3
# Number of bounded one-for-one exchange repair rounds.
MERGE_MAX_EXCHANGE_ROUNDS = 2
PROCESSING_RESULT_SUFFIX = rf"_可视化结果_{VALID_HOURS}.xlsx"
# 4. CH编号与对应实验参数的映射表
# 格式：CH编号: "Lp1-Np2-ICp3%" 完整参数字符串

# 0712
PARAM_MAPPING = {
    1: "L120‑N20‑IC5.25%",    # CH7_标准化.xlsx
    2: "L120‑N20‑IC5.25%",      # CH8_标准化.xlsx
    3: "L120‑N160‑IC5.25%",        # CH9_标准化.xlsx
    4: "L120‑N160‑IC5.25%",  # CH7_标准化.xlsx
    5: "L120‑N160‑IC5.25%",  # CH8_标准化.xlsx
    6: "L120‑N160‑IC5.25%",  # CH9_标准化.xlsx
    7: "L120‑N160‑IC0.5%",  # CH10_标准化.xlsx
    8: "L120‑N160‑IC0.5%",  # CH10_标准化.xlsx
    9: "L120‑N160‑IC10%",  # CH9_标准化.xlsx
    10: "L120‑N160‑IC0.5%",  # CH10_标准化.xlsx
    11: "L120‑N300‑IC5.25%",  # CH10_标准化.xlsx
    12: "L120‑N300‑IC5.25%",  # CH10_标准化.xlsx
    # 继续添加你所有的CH编号和对应参数字符串
}

# 0625
# PARAM_MAPPING = {
#     1: "L210-N20-IC10%-OC2",    # CH7_标准化.xlsx
#     2: "L210-N160-IC5.25%-OC2",      # CH8_标准化.xlsx
#     3: "L210-N300-IC0.5%-OC2",        # CH9_标准化.xlsx
#     4: "L210-N20-IC10%-OC2",  # CH7_标准化.xlsx
#     5: "L210-N160-IC5.25%-OC2",  # CH8_标准化.xlsx
#     6: "L210-N300-IC0.5%-OC2",  # CH9_标准化.xlsx
#     7: "L210-N20-IC10%-OC2",  # CH10_标准化.xlsx
#     8: "L210-N160-IC5.25%-OC2",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0504
# PARAM_MAPPING = {
#     1: "L120‑N160‑IC5.25%",    # CH7_标准化.xlsx
#     2: "L120‑N160‑IC5.25%",      # CH8_标准化.xlsx
#     3: "L120‑N20‑IC5.25%",        # CH9_标准化.xlsx
#     4: "L120‑N160‑IC5.25%",  # CH7_标准化.xlsx
#     5: "L120‑N160‑IC10%",  # CH8_标准化.xlsx
#     6: "L120‑N300‑IC5.25%",  # CH9_标准化.xlsx
#     7: "L210-N20-IC10%-OC2",  # CH10_标准化.xlsx
#     8: "L210-N160-IC5.25%-OC2",  # CH10_标准化.xlsx
#     9: "L30‑N300‑IC10%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0520
# PARAM_MAPPING = {
#     1: "L120‑N160‑IC0.5%",    # CH7_标准化.xlsx
#     2: "L120‑N160‑IC0.5%",      # CH8_标准化.xlsx
#     3: "L120‑N160‑IC5.25%",        # CH9_标准化.xlsx
#     4: "L30‑N20‑IC10%",  # CH7_标准化.xlsx
#     5: "L120‑N20‑IC5.25%",  # CH8_标准化.xlsx
#     6: "L120‑N300‑IC5.25%",  # CH9_标准化.xlsx
#     7: "L120‑N160‑IC5.25%",  # CH10_标准化.xlsx
#     8: "L120‑N160‑IC10%",  # CH10_标准化.xlsx
#     9: "L120‑N300‑IC5.25%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# # 0531
# PARAM_MAPPING = {
#     1: "L30‑N160‑IC5.25%",    # CH7_标准化.xlsx
#     2: "L30‑N160‑IC5.25%",      # CH8_标准化.xlsx
#     3: "L30‑N20‑IC0.5%",        # CH9_标准化.xlsx
#     4: "L30‑N20‑IC10%",  # CH7_标准化.xlsx
#     5: "L30‑N300‑IC0.5%",  # CH8_标准化.xlsx
#     6: "L30‑N300‑IC10%",  # CH9_标准化.xlsx
#     7: "L30‑N20‑IC0.5%",  # CH10_标准化.xlsx
#     8: "L30‑N300‑IC0.5%",  # CH10_标准化.xlsx
#     9: "L30‑N300‑IC10%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0609
# PARAM_MAPPING = {
#     1: "L210‑N160‑IC5.25%",    # CH7_标准化.xlsx
#     2: "L210‑N20‑IC0.5%",      # CH8_标准化.xlsx
#     3: "L210‑N20‑IC10%",        # CH9_标准化.xlsx
#     4: "L210‑N300‑IC0.5%",  # CH7_标准化.xlsx
#     5: "L210‑N300‑IC10%",  # CH8_标准化.xlsx
#     6: "L210‑N160‑IC5.25%",  # CH9_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0618
# PARAM_MAPPING = {
#     1: "L30‑N20‑IC0.5%-OC2",    # CH7_标准化.xlsx
#     2: "L30‑N300‑IC10%-OC2",      # CH8_标准化.xlsx
#     3: "L30‑N160‑IC5.25%-OC2",        # CH9_标准化.xlsx
#     4: "L30‑N20‑IC0.5%-OC2",  # CH7_标准化.xlsx
#     5: "L30‑N300‑IC10%-OC2",  # CH8_标准化.xlsx
#     6: "L30‑N160‑IC5.25%-OC2",  # CH9_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }
# ======================================================================================================


# 文件名匹配规则：提取CH后的数字编号
FILE_PATTERN = re.compile(r'^CH(\d{1,2})_标准化\.xlsx$')


# -------------------------- 单文件处理逻辑（完全对齐原代码调用） --------------------------
def process_single_ch(ch_num: int, input_path: str, output_path: str):
    """
    处理单个CH的Excel文件，调用原有的process_cell_growth函数
    """
    # 从映射表取出对应参数字符串
    experimental_params = PARAM_MAPPING[ch_num]
    print(f"\n✅ 正在处理 CH{ch_num}，实验参数：{experimental_params}")

    # 完全按照原代码的参数名和格式调用函数
    result = process_cell_growth(
        excel_path=input_path,
        result_excel_path=output_path,
        min_data_points=MIN_DATA_POINTS,
        experimental_parameters=experimental_params,
        skip_sheet_name=SKIP_SHEET,
        valid_num=SET_VALID_NUM,
    )

    # 处理结果判断和预览（和原代码输出逻辑一致）
    if result is not None:
        print(f"\n📊 CH{ch_num} 拟合结果预览（含生长阶段时长）：")
        print(result[["腔室名称", "滞止期参数λ(h)", "滞后期时长(h)", "对数期时长(h)", "稳定期时长(h)", "稳定期起始t90(h)"]].head())
        return True
    else:
        print(f"❌ CH{ch_num} 处理失败，返回结果为空")
        return False


# -------------------------- 批量主程序 --------------------------
if __name__ == "__main__":
    # 1. 自动创建输出文件夹（不存在则新建）
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("=" * 60)
    print(f"📂 输入文件夹：{INPUT_FOLDER}")
    print(f"📂 输出文件夹：{OUTPUT_FOLDER}")
    print(f"⚙️  全局配置：最小数据点={MIN_DATA_POINTS}，有效数量={SET_VALID_NUM}，跳过工作表={SKIP_SHEET}")
    print("=" * 60)

    # 2. 扫描所有符合命名规则的Excel文件
    file_list = []
    for filename in os.listdir(INPUT_FOLDER):
        match = FILE_PATTERN.match(filename)
        if match:
            ch_num = int(match.group(1))
            input_full_path = os.path.join(INPUT_FOLDER, filename)
            file_list.append((ch_num, filename, input_full_path))

    if not file_list:
        print("\n❌ 未找到符合格式的Excel文件（CH数字_标准化.xlsx）")
    else:
        print(f"\n✅ 共扫描到 {len(file_list)} 个待处理文件，开始批量处理...")
        print("-" * 60)

        success_count = 0
        # 3. 循环逐个处理
        for idx, (ch_num, filename, input_path) in enumerate(file_list, 1):
            print(f"\n[{idx}/{len(file_list)}] 文件：{filename}")

            # 检查是否在参数映射表中
            if ch_num not in PARAM_MAPPING:
                print(f"⚠️  CH{ch_num} 未配置对应实验参数，跳过该文件")
                continue

            # 生成输出文件路径：CH7_标准化.xlsx → CH7_可视化结果.xlsx（和原代码命名一致）
            output_filename = filename.replace("_标准化.xlsx", PROCESSING_RESULT_SUFFIX)
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # 执行单文件处理
            is_success = process_single_ch(ch_num, input_path, output_path)
            if is_success:
                success_count += 1

        # 4. 处理结束统计
        print("\n" + "=" * 60)
        print(f"🎉 批量处理全部完成")
        print(f"   总文件数：{len(file_list)} 个")
        print(f"   成功处理：{success_count} 个")
        print(f"   跳过/失败：{len(file_list) - success_count} 个")
        print(f"   所有结果已保存至：{OUTPUT_FOLDER}")
        print("=" * 60)
