"""Batch growth-kinetics analysis for standardized microalgae workbooks.

The script keeps source workbooks untouched and writes one analysis workbook
per input file. It performs point-level QC, chamber-level fixed-N0 logistic
fits, robust chamber selection, channel aggregation, and chamber bootstrap.

Example:
    python growth_kinetics_pipeline.py \
        --input-dir "F:\\Microalgae_Photoes\\Archive\\20260504\\数据汇总\\02_标准化数据" \
        --output-dir "F:\\Microalgae_Photoes\\Archive\\20260504\\数据汇总\\03_生长动力学" \
        --time-step 1 --start-time 0 --bootstrap 500

本脚本旨在对标准化的微藻生长数据工作簿进行批量生长动力学分析。它保持源工作簿不变，并为每个输入文件生成一个独立的分析结果工作簿。
主要处理流程包括：
点级质控 (Point-level QC)：识别并标记异常数据点。
腔室级拟合 (Chamber-level fitting)：对每个腔室的数据进行固定初始值（N0=1）的 Logistic 模型拟合。
腔室筛选 (Chamber selection)：根据拟合优度（R²）等标准筛选出可靠的腔室。
流道聚合 (Channel aggregation)：将筛选后的腔室数据聚合，生成代表整个流道的生长曲线。
Bootstrap 分析：通过重采样方法评估 μmax（最大比生长速率）等参数的不确定性。
"""
from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import trim_mean


# ================================== 【配置区：所有可修改参数都在这里】 ==================================
DATE_STR = "20260726"
VALID_HOURS = "96"

# 1. 输入和输出文件夹。修改 DATE_STR 后，下面两个路径会自动更新。
INPUT_FOLDER = rf"F:\Microalgae_Photoes\Test\{DATE_STR}\数据汇总\02_标准化数据"
OUTPUT_FOLDER = rf"F:\Microalgae_Photoes\Test\{DATE_STR}\数据汇总\04_生长动力学结果\{VALID_HOURS}小时"

# 2. 文件和时间设置。
SKIP_SHEET = {"数据汇总"}
SET_VALID_NUM = int(VALID_HOURS) + 1  # 0 h 到 VALID_HOURS，共 VALID_HOURS+1 个点
TIME_START_H = 0.0
TIME_STEP_H = 1.0
DEFAULT_FILE_PATTERN = r"^CH\d{1,2}_标准化\.xlsx$"
PROCESSING_RESULT_SUFFIX = rf"_生长动力学分析_{VALID_HOURS}小时.xlsx"

# 3. 质控和拟合参数。
MIN_DATA_POINTS = 5
MIN_VALID_FRACTION = 0.80
MIN_R2 = 0.90
MIN_CHAMBERS = 3
AGGREGATE_METHOD = "median"  # 可选：median、mean、trimmed_mean
TRIM_FRACTION = 0.10
POINT_ROBUST_Z = 3.5
POINT_SCALE_FLOOR = 0.05
MAX_FOLD_CHANGE = 4.0
EXCLUDE_POINT_JUMPS = False  # False：只记录突变；True：将突变点排除出拟合
MU_MAD_MULTIPLIER = 3.0
KEEP_MU_OUTLIERS = False
MAX_MU = 2.0
K_UPPER_FACTOR = 20.0
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20260819

# 4. CH编号与实验参数映射。缺少映射的文件仍会处理，但会标记为“未配置”。
# 格式：CH编号: "Lp1-Np2-ICp3%" 完整参数字符串

# 0726
PARAM_MAPPING = {
    1: "L30‑N160‑IC5.25%",    # CH7_标准化.xlsx
    2: "L30‑N160‑IC5.25%",      # CH8_标准化.xlsx
    3: "L30‑N20‑IC10%",        # CH9_标准化.xlsx
    4: "L30‑N20‑IC0.5%",  # CH7_标准化.xlsx
    5: "L30‑N20‑IC0.5%",  # CH8_标准化.xlsx
    6: "LL30‑N20‑IC0.5%",  # CH9_标准化.xlsx
    7: "L30‑N20‑IC10%",  # CH10_标准化.xlsx
    8: "L30‑N300‑IC0.5%",  # CH10_标准化.xlsx
    9: "L30‑N300‑IC0.5%",  # CH9_标准化.xlsx
    10: "L30‑N300‑IC10%",  # CH10_标准化.xlsx
    11: "L30‑N300‑IC10%",  # CH10_标准化.xlsx
    12: "L30‑N300‑IC10%",  # CH10_标准化.xlsx
    # 继续添加你所有的CH编号和对应参数字符串
}

# 0719
# PARAM_MAPPING = {
#     1: "L210-N20-IC0.5%",    # CH7_标准化.xlsx
#     2: "L210-N20-IC0.5%",      # CH8_标准化.xlsx
#     3: "L210-N20-IC0.5%",        # CH9_标准化.xlsx
#     4: "L210-N160-IC5.25%",  # CH7_标准化.xlsx
#     5: "L210-N160-IC5.25%",  # CH8_标准化.xlsx
#     6: "L210-N300-IC0.5%",  # CH9_标准化.xlsx
#     7: "L210-N300-IC0.5%",  # CH10_标准化.xlsx
#     8: "L210-N300-IC10%",  # CH10_标准化.xlsx
#     9: "L210-N300-IC10%",  # CH9_标准化.xlsx
#     10: "L210-N20-IC10%",  # CH10_标准化.xlsx
#     11: "L210-N20-IC10%",  # CH10_标准化.xlsx
#     12: "L210-N20-IC10%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0712
# PARAM_MAPPING = {
#     1: "L120-N20-IC5.25%",    # CH7_标准化.xlsx
#     2: "L120-N20-IC5.25%",      # CH8_标准化.xlsx
#     3: "L120-N160-IC5.25%",        # CH9_标准化.xlsx
#     4: "L120-N160-IC5.25%",  # CH7_标准化.xlsx
#     5: "L120-N160-IC5.25%",  # CH8_标准化.xlsx
#     6: "L120-N160-IC5.25%",  # CH9_标准化.xlsx
#     7: "L120-N160-IC0.5%",  # CH10_标准化.xlsx
#     8: "L120-N160-IC0.5%",  # CH10_标准化.xlsx
#     9: "L120-N160-IC10%",  # CH9_标准化.xlsx
#     10: "L120-N160-IC0.5%",  # CH10_标准化.xlsx
#     11: "L120-N300-IC5.25%",  # CH10_标准化.xlsx
#     12: "L120-N300-IC5.25%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

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
#     1: "L120-N160-IC5.25%",    # CH7_标准化.xlsx
#     2: "L120-N160-IC5.25%",      # CH8_标准化.xlsx
#     3: "L120-N20-IC5.25%",        # CH9_标准化.xlsx
#     4: "L120-N160-IC5.25%",  # CH7_标准化.xlsx
#     5: "L120-N160-IC10%",  # CH8_标准化.xlsx
#     6: "L120-N300-IC5.25%",  # CH9_标准化.xlsx
#     7: "L210-N20-IC10%-OC2",  # CH10_标准化.xlsx
#     8: "L210-N160-IC5.25%-OC2",  # CH10_标准化.xlsx
#     9: "L30-N300-IC10%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0520
# PARAM_MAPPING = {
#     1: "L120-N160-IC0.5%",    # CH7_标准化.xlsx
#     2: "L120-N160-IC0.5%",      # CH8_标准化.xlsx
#     3: "L120-N160-IC5.25%",        # CH9_标准化.xlsx
#     4: "L30-N20-IC10%",  # CH7_标准化.xlsx
#     5: "L120-N20-IC5.25%",  # CH8_标准化.xlsx
#     6: "L120-N300-IC5.25%",  # CH9_标准化.xlsx
#     7: "L120-N160-IC5.25%",  # CH10_标准化.xlsx
#     8: "L120-N160-IC10%",  # CH10_标准化.xlsx
#     9: "L120-N300-IC5.25%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0531
# PARAM_MAPPING = {
#     1: "L30-N160-IC5.25%",    # CH7_标准化.xlsx
#     2: "L30-N160-IC5.25%",      # CH8_标准化.xlsx
#     3: "L30-N20-IC0.5%",        # CH9_标准化.xlsx
#     4: "L30-N20-IC10%",  # CH7_标准化.xlsx
#     5: "L30-N300-IC0.5%",  # CH8_标准化.xlsx
#     6: "L30-N300-IC10%",  # CH9_标准化.xlsx
#     7: "L30-N20-IC0.5%",  # CH10_标准化.xlsx
#     8: "L30-N300-IC0.5%",  # CH10_标准化.xlsx
#     9: "L30-N300-IC10%",  # CH10_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0609
# PARAM_MAPPING = {
#     1: "L210-N160-IC5.25%",    # CH7_标准化.xlsx
#     2: "L210-N20-IC0.5%",      # CH8_标准化.xlsx
#     3: "L210-N20-IC10%",        # CH9_标准化.xlsx
#     4: "L210-N300-IC0.5%",  # CH7_标准化.xlsx
#     5: "L210-N300-IC10%",  # CH8_标准化.xlsx
#     6: "L210-N160-IC5.25%",  # CH9_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }

# 0618
# PARAM_MAPPING = {
#     1: "L30-N20-IC0.5%-OC2",    # CH7_标准化.xlsx
#     2: "L30-N300-IC10%-OC2",      # CH8_标准化.xlsx
#     3: "L30-N160-IC5.25%-OC2",        # CH9_标准化.xlsx
#     4: "L30-N20-IC0.5%-OC2",  # CH7_标准化.xlsx
#     5: "L30-N300-IC10%-OC2",  # CH8_标准化.xlsx
#     6: "L30-N160-IC5.25%-OC2",  # CH9_标准化.xlsx
#     # 继续添加你所有的CH编号和对应参数字符串
# }
# =====================================================================================================

SUMMARY_SHEETS = set(SKIP_SHEET) | {"汇总结果", "Summary", "summary"}

COLUMN_ALIASES = {
    "image_name": ["图片名称", "图像名称", "image_name", "filename"],
    "status": ["处理状态", "状态", "status"],
    "target_count": ["目标数量", "细胞数量", "目标数", "cell_count"],
    "total_area": ["总面积(μm²)", "总面积(um²)", "总面积", "total_area"],
    "avg_cell_area": [
        "相对平均细胞面积",
        "平均细胞面积",
        "relative_average_cell_area",
    ],
    "time": ["时间", "培养时间", "培养时间(h)", "time", "time_h"],
}

# 仅在写入 Excel/CSV 时替换表头，内部计算继续使用稳定的英文键名。
OUTPUT_HEADER_CN = {
    "source_file": "源文件",
    "channel": "流道",
    "chamber": "腔室",
    "image_name": "图片名称",
    "time_h": "培养时间(h)",
    "status": "处理状态",
    "target_count": "相对目标数量",
    "total_area": "相对总面积",
    "avg_cell_area": "相对平均细胞面积",
    "robust_z": "稳健Z值",
    "status_ok": "处理状态合格",
    "numeric_ok": "数值有效",
    "point_robust_outlier": "点级稳健异常",
    "log_fold_change": "相邻点对数变化",
    "point_jump": "相邻点突变",
    "point_qc_flag": "点级质控标记",
    "point_valid": "点级基础有效",
    "point_include": "参与拟合",
    "total_points": "总时间点数",
    "valid_points": "有效时间点数",
    "valid_fraction": "有效时间点比例",
    "point_outlier_count": "异常点数量",
    "mu_max": "最大比生长速率μmax(h^-1)",
    "k": "环境容纳量K",
    "r2": "拟合优度R²",
    "rmse": "均方根误差RMSE",
    "relative_rmse": "相对RMSE",
    "doubling_time_h": "倍增时间(h)",
    "mu_exp": "指数期比生长速率μ(h^-1)",
    "r2_exp": "指数期拟合优度R²",
    "k_at_upper": "K接近参数上限",
    "fit_status": "拟合状态",
    "quality_flags": "质量标记",
    "start_index": "指数期起始点序号",
    "end_index": "指数期结束点序号",
    "mu_median": "腔室μmax中位数",
    "mu_mad": "腔室μmax的MAD",
    "mu_robust_outlier": "μmax稳健异常",
    "n_chambers": "有效腔室数",
    "mean": "平均值",
    "median": "中位数",
    "trimmed_mean": "截尾平均值",
    "std": "标准差",
    "min": "最小值",
    "max": "最大值",
    "experimental_parameters": "实验参数",
    "total_chambers": "总腔室数",
    "accepted_chambers": "合格腔室数",
    "accepted_chamber_names": "参与拟合的腔室",
    "aggregate_method": "流道聚合方法",
    "primary_mu_max": "流道μmax(h^-1)",
    "primary_k": "流道环境容纳量K",
    "primary_r2": "流道拟合优度R²",
    "primary_rmse": "流道拟合RMSE",
    "primary_doubling_time_h": "流道倍增时间(h)",
    "bootstrap_median_mu": "Bootstrap μmax中位数",
    "bootstrap_ci_low": "Bootstrap 95%CI下限",
    "bootstrap_ci_high": "Bootstrap 95%CI上限",
    "bootstrap_success": "Bootstrap成功次数",
    "all_chambers_median_curve_mu": "全部腔室中位曲线μmax(h^-1)",
    "level": "质控层级",
    "reason": "原因",
    "action": "处理方式",
    "parameter": "参数名",
    "value": "参数值",
}

AGGREGATE_LABELS_CN = {
    "mean": "平均值",
    "median": "中位数",
    "trimmed_mean": "截尾平均值",
}


def normalize_header(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).strip().lower()


def find_column(columns: Iterable[object], aliases: list[str]) -> object | None:
    normalized = {normalize_header(col): col for col in columns}
    for alias in aliases:
        key = normalize_header(alias)
        if key in normalized:
            return normalized[key]
    for col in columns:
        key = normalize_header(col)
        if any(normalize_header(alias) in key for alias in aliases):
            return col
    return None


def as_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def status_is_success(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"成功", "success", "ok", "通过", "valid"} or "成功" in text


def fixed_n0_logistic(t: np.ndarray, mu_max: float, k: float) -> np.ndarray:
    """Logistic curve for data normalized to N(0)=1."""
    t = np.asarray(t, dtype=float)
    exponent = np.clip(-mu_max * t, -700.0, 700.0)
    return k / (1.0 + (k - 1.0) * np.exp(exponent))


def fit_fixed_n0_logistic(
    t: np.ndarray,
    y: np.ndarray,
    max_mu: float,
    k_upper_factor: float,
) -> dict[str, float | bool | np.ndarray]:
    if len(t) < 5:
        raise ValueError("有效数据点少于5个")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("拟合数据包含缺失、非有限值或非正数")

    max_y = max(float(np.max(y)), 1.0)
    k_upper = max(2.0, max_y * k_upper_factor)
    p0 = [min(0.15, max_mu * 0.5), max(1.05, max_y * 1.1)]
    p0 = np.clip(p0, [1e-6, 1.000001], [max_mu * 0.999999, k_upper * 0.95])

    params, _ = curve_fit(
        fixed_n0_logistic,
        t,
        y,
        p0=p0,
        bounds=([1e-6, 1.000001], [max_mu, k_upper]),
        maxfev=30000,
    )
    mu_max, k = map(float, params)
    y_fit = fixed_n0_logistic(t, mu_max, k)
    residuals = y - y_fit
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    relative_rmse = rmse / max(float(np.mean(np.abs(y))), 1e-12)
    return {
        "mu_max": mu_max,
        "k": k,
        "r2": float(r2),
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "doubling_time_h": float(np.log(2.0) / mu_max),
        "k_upper": k_upper,
        "k_at_upper": bool(k >= k_upper * 0.995),
        "y_fit": y_fit,
    }


def fit_log_linear_growth(t: np.ndarray, y: np.ndarray, min_points: int) -> dict[str, float | int] | None:
    """Fit the longest increasing log-linear window as a diagnostic."""
    if len(t) < min_points or (y <= 0).any():
        return None
    log_y = np.log(y)
    best: dict[str, float | int] | None = None
    min_window = max(min_points, int(math.ceil(len(t) * 0.2)))
    for start in range(0, len(t) - min_window + 1):
        for end in range(start + min_window, len(t) + 1):
            x = t[start:end]
            z = log_y[start:end]
            slope, intercept = np.polyfit(x, z, 1)
            if slope <= 0:
                continue
            fitted = slope * x + intercept
            ss_res = float(np.sum((z - fitted) ** 2))
            ss_tot = float(np.sum((z - np.mean(z)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            candidate = {"mu_exp": float(slope), "r2_exp": float(r2), "start_index": start, "end_index": end - 1}
            if best is None:
                best = candidate
                continue
            best_length = int(best["end_index"]) - int(best["start_index"])
            candidate_length = end - 1 - start
            if candidate_length > best_length or (
                candidate_length == best_length and float(candidate["r2_exp"]) > float(best["r2_exp"])
            ):
                best = candidate
    return best


def robust_z_by_time(points: pd.DataFrame, value_col: str, floor_fraction: float) -> pd.Series:
    result = pd.Series(np.nan, index=points.index, dtype=float)
    for _, index in points.groupby("time_h", sort=False).groups.items():
        values = points.loc[index, value_col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values[finite] - median)))
        scale = max(1.4826 * mad, floor_fraction * max(abs(median), 1.0), 1e-9)
        result.loc[index] = np.abs(values - median) / scale
    return result


def add_point_qc(
    points: pd.DataFrame,
    robust_z_threshold: float,
    floor_fraction: float,
    max_fold_change: float,
    exclude_point_jumps: bool,
) -> pd.DataFrame:
    points = points.copy()
    points["robust_z"] = robust_z_by_time(points, "target_count", floor_fraction)
    points["status_ok"] = points["status"].map(status_is_success)
    points["numeric_ok"] = np.isfinite(points["target_count"]) & (points["target_count"] > 0)
    points["point_robust_outlier"] = points["robust_z"] > robust_z_threshold
    points["log_fold_change"] = np.nan
    points["point_jump"] = False

    for chamber, index in points.groupby("chamber", sort=False).groups.items():
        ordered = points.loc[index].sort_values("time_h")
        y = ordered["target_count"].to_numpy(dtype=float)
        log_y = np.log(np.where(y > 0, y, np.nan))
        fold_change = np.full(len(ordered), np.nan)
        fold_change[1:] = np.abs(log_y[1:] - log_y[:-1])
        jump = fold_change > np.log(max_fold_change)
        points.loc[ordered.index, "log_fold_change"] = fold_change
        points.loc[ordered.index, "point_jump"] = np.nan_to_num(jump, nan=False)

    flag_columns = {
        "status_ok": "状态非成功",
        "numeric_ok": "目标数量非正或缺失",
        "point_robust_outlier": "跨腔室稳健异常",
        "point_jump": "相邻时间点突变",
    }
    flags: list[str] = []
    for _, row in points.iterrows():
        row_flags = [label for key, label in flag_columns.items() if not bool(row[key])]
        if bool(row["point_robust_outlier"]):
            row_flags.append("跨腔室稳健异常")
        if bool(row["point_jump"]):
            row_flags.append("相邻时间点突变")
        flags.append(";".join(dict.fromkeys(row_flags)))
    points["point_qc_flag"] = flags
    points["point_valid"] = points["status_ok"] & points["numeric_ok"]
    points["point_include"] = points["point_valid"] & ~points["point_robust_outlier"]
    if exclude_point_jumps:
        points["point_include"] &= ~points["point_jump"]
    return points


def fit_chambers(
    points: pd.DataFrame,
    min_points: int,
    min_valid_fraction: float,
    min_r2: float,
    max_mu: float,
    k_upper_factor: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chamber, group in points.groupby("chamber", sort=False):
        fitting = group[group["point_include"]].sort_values("time_h")
        valid_count = len(fitting)
        total_count = len(group)
        row: dict[str, object] = {
            "chamber": chamber,
            "total_points": total_count,
            "valid_points": valid_count,
            "valid_fraction": valid_count / total_count if total_count else 0.0,
            "point_outlier_count": int((group["point_valid"] & ~group["point_include"]).sum()),
            "mu_max": np.nan,
            "k": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "relative_rmse": np.nan,
            "doubling_time_h": np.nan,
            "mu_exp": np.nan,
            "r2_exp": np.nan,
            "k_at_upper": False,
            "fit_status": "excluded",
            "quality_flags": "",
        }
        flags: list[str] = []
        rejection_flags: list[str] = []
        warning_flags: list[str] = []
        if valid_count < min_points:
            rejection_flags.append("有效点数不足")
        if row["valid_fraction"] < min_valid_fraction:
            rejection_flags.append("有效点比例不足")
        if rejection_flags:
            row["quality_flags"] = ";".join(rejection_flags)
            rows.append(row)
            continue
        try:
            t = fitting["time_h"].to_numpy(dtype=float)
            y = fitting["target_count"].to_numpy(dtype=float)
            fit = fit_fixed_n0_logistic(t, y, max_mu, k_upper_factor)
            row.update(
                {
                    "mu_max": fit["mu_max"],
                    "k": fit["k"],
                    "r2": fit["r2"],
                    "rmse": fit["rmse"],
                    "relative_rmse": fit["relative_rmse"],
                    "doubling_time_h": fit["doubling_time_h"],
                    "k_at_upper": fit["k_at_upper"],
                }
            )
            exp_fit = fit_log_linear_growth(t, y, min_points)
            if exp_fit:
                row.update(exp_fit)
            if not np.isfinite(float(fit["r2"])) or float(fit["r2"]) < min_r2:
                rejection_flags.append(f"R²低于{min_r2:g}")
            if bool(fit["k_at_upper"]):
                warning_flags.append("K接近参数上限")
            if row["point_outlier_count"]:
                warning_flags.append("包含被排除的异常点")
            flags = rejection_flags + warning_flags
            row["fit_status"] = "excluded" if rejection_flags else ("accepted_with_warning" if warning_flags else "accepted")
        except Exception as exc:
            rejection_flags.append(f"拟合失败:{exc}")
            flags = rejection_flags
        row["quality_flags"] = ";".join(dict.fromkeys(flags))
        rows.append(row)
    return pd.DataFrame(rows)


def mark_mu_outliers(chamber_fit: pd.DataFrame, keep_mu_outliers: bool, mad_multiplier: float) -> pd.DataFrame:
    result = chamber_fit.copy()
    successful = result["mu_max"].notna() & result["r2"].notna()
    mus = result.loc[successful, "mu_max"].to_numpy(dtype=float)
    result["mu_median"] = np.nan
    result["mu_mad"] = np.nan
    result["mu_robust_outlier"] = False
    if len(mus) >= 3:
        median = float(np.median(mus))
        mad = float(np.median(np.abs(mus - median)))
        scale = max(1.4826 * mad, 1e-9)
        threshold = mad_multiplier * scale
        outlier = successful & (np.abs(result["mu_max"] - median) > threshold)
        result.loc[:, "mu_median"] = median
        result.loc[:, "mu_mad"] = mad
        result.loc[outlier, "mu_robust_outlier"] = True
        if not keep_mu_outliers:
            result.loc[outlier, "fit_status"] = "excluded"
            result.loc[outlier, "quality_flags"] = result.loc[outlier, "quality_flags"].map(
                lambda value: ";".join(dict.fromkeys(filter(None, [value, "μmax稳健异常"])))
            )
    return result


def aggregate_channel(
    points: pd.DataFrame,
    accepted_chambers: list[str],
    trim_fraction: float,
) -> pd.DataFrame:
    selected = points[points["chamber"].isin(accepted_chambers) & points["point_include"]]
    rows: list[dict[str, object]] = []
    for time_h, group in selected.groupby("time_h", sort=True):
        values = group["target_count"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        trimmed = float(trim_mean(values, proportiontocut=trim_fraction)) if len(values) >= 4 else float(np.mean(values))
        rows.append(
            {
                "time_h": float(time_h),
                "n_chambers": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "trimmed_mean": trimmed,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def fit_aggregate_curve(aggregate: pd.DataFrame, method: str, min_chambers: int, max_mu: float, k_upper_factor: float) -> dict[str, object] | None:
    if aggregate.empty:
        return None
    usable = aggregate[aggregate["n_chambers"] >= min_chambers].copy()
    y_col = {"mean": "mean", "median": "median", "trimmed_mean": "trimmed_mean"}[method]
    usable = usable[np.isfinite(usable[y_col]) & (usable[y_col] > 0)]
    if len(usable) < 5:
        return None
    fit = fit_fixed_n0_logistic(
        usable["time_h"].to_numpy(dtype=float),
        usable[y_col].to_numpy(dtype=float),
        max_mu,
        k_upper_factor,
    )
    return {"method": method, "n_timepoints": len(usable), **fit}


def create_summary_fit_chart(
    aggregate: pd.DataFrame,
    primary_fit: dict[str, object],
    aggregate_method: str,
    channel: str,
    experimental_parameters: str,
    accepted_chambers: int,
) -> io.BytesIO:
    """生成汇总页使用的流道代表曲线与Logistic拟合图。"""
    y_column = {
        "mean": "mean",
        "median": "median",
        "trimmed_mean": "trimmed_mean",
    }[aggregate_method]
    usable = aggregate[
        np.isfinite(aggregate[y_column]) & (aggregate[y_column] > 0)
    ].copy()

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(
        usable["time_h"],
        usable[y_column],
        color="#1F77B4",
        s=34,
        alpha=0.78,
        label=f"流道{AGGREGATE_LABELS_CN[aggregate_method]}数据",
        zorder=3,
    )

    dense_time = np.linspace(
        float(usable["time_h"].min()),
        float(usable["time_h"].max()),
        500,
    )
    fitted_curve = fixed_n0_logistic(
        dense_time,
        float(primary_fit["mu_max"]),
        float(primary_fit["k"]),
    )
    ax.plot(
        dense_time,
        fitted_curve,
        color="#D62728",
        linewidth=2.4,
        label="Logistic拟合曲线",
        zorder=4,
    )

    ax.set_xlabel("培养时间 (h)", fontsize=13)
    ax.set_ylabel("相对细胞数量（以0 h标准化）", fontsize=13)
    ax.set_title(f"{channel} 流道生长曲线拟合", fontsize=16, pad=12)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(3))
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True)

    parameter_text = (
        f"实验参数：{experimental_parameters}\n"
        f"参与拟合腔室：{accepted_chambers}\n"
        f"μmax = {float(primary_fit['mu_max']):.4f} h^-1\n"
        f"K = {float(primary_fit['k']):.2f}\n"
        f"R² = {float(primary_fit['r2']):.4f}\n"
        f"RMSE = {float(primary_fit['rmse']):.4f}"
    )
    ax.text(
        0.025,
        0.965,
        parameter_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#9E9E9E", "alpha": 0.90},
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


def bootstrap_mu(
    points: pd.DataFrame,
    accepted_chambers: list[str],
    aggregate_method: str,
    trim_fraction: float,
    min_chambers: int,
    max_mu: float,
    k_upper_factor: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    if n_bootstrap <= 0 or len(accepted_chambers) < min_chambers:
        return np.nan, np.nan, np.nan, 0
    pivot = points[points["point_include"] & points["chamber"].isin(accepted_chambers)].pivot_table(
        index="time_h", columns="chamber", values="target_count", aggfunc="mean"
    )
    rng = np.random.default_rng(seed)
    mus: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(accepted_chambers, size=len(accepted_chambers), replace=True)
        sampled = pivot.reindex(columns=sample)
        values = sampled.to_numpy(dtype=float)
        counts = np.sum(np.isfinite(values), axis=1)
        with np.errstate(all="ignore"):
            if aggregate_method == "mean":
                curve = np.nanmean(values, axis=1)
            elif aggregate_method == "trimmed_mean":
                curve = np.array(
                    [trim_mean(row[np.isfinite(row)], proportiontocut=trim_fraction) if np.isfinite(row).sum() >= 4 else np.nan for row in values]
                )
            else:
                curve = np.nanmedian(values, axis=1)
        usable = np.isfinite(curve) & (counts >= min_chambers) & (curve > 0)
        if usable.sum() < 5:
            continue
        try:
            fit = fit_fixed_n0_logistic(pivot.index.to_numpy(dtype=float)[usable], curve[usable], max_mu, k_upper_factor)
            mus.append(float(fit["mu_max"]))
        except Exception:
            continue
    if not mus:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(mus)), float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5)), len(mus)


def write_dataframe_sheet(workbook: Workbook, title: str, dataframe: pd.DataFrame):
    output_dataframe = dataframe.rename(columns=OUTPUT_HEADER_CN)
    ws = workbook.create_sheet(title=title[:31])
    ws.freeze_panes = "A2"
    ws.sheet_view.showGridLines = False
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for col_idx, column in enumerate(output_dataframe.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=str(column))
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row_idx, row in enumerate(output_dataframe.itertuples(index=False, name=None), start=2):
        for col_idx, value in enumerate(row, start=1):
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            if isinstance(value, float):
                cell.number_format = "0.0000"
    ws.auto_filter.ref = ws.dimensions
    for col_idx, column in enumerate(output_dataframe.columns, start=1):
        values = [str(column)] + [str(value) for value in output_dataframe.iloc[:, col_idx - 1].head(100) if pd.notna(value)]
        width = min(max(max((len(value) for value in values), default=10) + 2, 10), 32)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    return ws


def build_qc_log(points: pd.DataFrame, chamber_fit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in points[points["point_qc_flag"] != ""].iterrows():
        rows.append(
            {
                "level": "point",
                "chamber": row["chamber"],
                "time_h": row["time_h"],
                "image_name": row["image_name"],
                "reason": row["point_qc_flag"],
                "action": "excluded_from_fit" if not row["point_include"] else "flag_only",
            }
        )
    for _, row in chamber_fit.iterrows():
        if row["quality_flags"]:
            rows.append(
                {
                    "level": "chamber",
                    "chamber": row["chamber"],
                    "time_h": None,
                    "image_name": None,
                    "reason": row["quality_flags"],
                    "action": row["fit_status"],
                }
            )
    return pd.DataFrame(rows, columns=["level", "chamber", "time_h", "image_name", "reason", "action"])


def read_workbook_points(path: Path, start_time: float, time_step: float, max_points: int | None) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        if sheet in SUMMARY_SHEETS:
            continue
        dataframe = pd.read_excel(path, sheet_name=sheet)
        if max_points is not None:
            dataframe = dataframe.iloc[:max_points].copy()
        if dataframe.empty:
            continue
        count_col = find_column(dataframe.columns, COLUMN_ALIASES["target_count"])
        if count_col is None:
            continue
        image_col = find_column(dataframe.columns, COLUMN_ALIASES["image_name"])
        status_col = find_column(dataframe.columns, COLUMN_ALIASES["status"])
        area_col = find_column(dataframe.columns, COLUMN_ALIASES["total_area"])
        avg_area_col = find_column(dataframe.columns, COLUMN_ALIASES["avg_cell_area"])
        time_col = find_column(dataframe.columns, COLUMN_ALIASES["time"])

        count_values = as_float_series(dataframe[count_col])
        for index, value in enumerate(count_values):
            if time_col is not None:
                time_value = pd.to_numeric(pd.Series([dataframe.iloc[index][time_col]]), errors="coerce").iloc[0]
                time_value = float(time_value) if pd.notna(time_value) else start_time + index * time_step
            else:
                time_value = start_time + index * time_step
            records.append(
                {
                    "source_file": path.name,
                    "channel": re.match(r"^(CH\d+)", path.stem).group(1) if re.match(r"^(CH\d+)", path.stem) else path.stem,
                    "chamber": str(sheet),
                    "image_name": dataframe.iloc[index][image_col] if image_col is not None else f"{sheet}_{index + 1}",
                    "time_h": time_value,
                    "status": dataframe.iloc[index][status_col] if status_col is not None else "",
                    "target_count": value,
                    "total_area": dataframe.iloc[index][area_col] if area_col is not None else np.nan,
                    "avg_cell_area": dataframe.iloc[index][avg_area_col] if avg_area_col is not None else np.nan,
                }
            )
    if not records:
        raise ValueError("没有找到包含目标数量列的腔室工作表")
    return pd.DataFrame(records)


def process_workbook(
    path: Path,
    output_path: Path,
    args: argparse.Namespace,
    experimental_parameters: str,
) -> dict[str, object]:
    points = read_workbook_points(path, args.start_time, args.time_step, args.max_points)
    points = add_point_qc(
        points,
        args.point_robust_z,
        args.point_scale_floor,
        args.max_fold_change,
        args.exclude_point_jumps,
    )
    chamber_fit = fit_chambers(
        points,
        args.min_points,
        args.min_valid_fraction,
        args.min_r2,
        args.max_mu,
        args.k_upper_factor,
    )
    chamber_fit = mark_mu_outliers(chamber_fit, args.keep_mu_outliers, args.mu_mad_multiplier)
    accepted = chamber_fit[chamber_fit["fit_status"].isin(["accepted", "accepted_with_warning"])]
    accepted_chambers = accepted["chamber"].astype(str).tolist()
    aggregate = aggregate_channel(points, accepted_chambers, args.trim_fraction)
    primary_fit = fit_aggregate_curve(
        aggregate,
        args.aggregate,
        args.min_chambers,
        args.max_mu,
        args.k_upper_factor,
    )
    bootstrap_median, bootstrap_low, bootstrap_high, bootstrap_success = bootstrap_mu(
        points,
        accepted_chambers,
        args.aggregate,
        args.trim_fraction,
        args.min_chambers,
        args.max_mu,
        args.k_upper_factor,
        args.bootstrap,
        args.seed,
    )

    all_fit = fit_aggregate_curve(
        aggregate_channel(points, chamber_fit[chamber_fit["mu_max"].notna()]["chamber"].astype(str).tolist(), args.trim_fraction),
        "median",
        args.min_chambers,
        args.max_mu,
        args.k_upper_factor,
    )
    channel = points["channel"].iloc[0]
    summary = {
        "source_file": path.name,
        "channel": channel,
        "experimental_parameters": experimental_parameters,
        "total_chambers": int(chamber_fit.shape[0]),
        "accepted_chambers": len(accepted_chambers),
        "accepted_chamber_names": ", ".join(accepted_chambers),
        "aggregate_method": args.aggregate,
        "primary_mu_max": primary_fit["mu_max"] if primary_fit else np.nan,
        "primary_k": primary_fit["k"] if primary_fit else np.nan,
        "primary_r2": primary_fit["r2"] if primary_fit else np.nan,
        "primary_rmse": primary_fit["rmse"] if primary_fit else np.nan,
        "primary_doubling_time_h": primary_fit["doubling_time_h"] if primary_fit else np.nan,
        "bootstrap_median_mu": bootstrap_median,
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
        "bootstrap_success": bootstrap_success,
        "all_chambers_median_curve_mu": all_fit["mu_max"] if all_fit else np.nan,
        "status": "ok" if primary_fit else "no_channel_fit",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    write_dataframe_sheet(workbook, "点级数据", points)
    write_dataframe_sheet(workbook, "腔室拟合", chamber_fit)
    write_dataframe_sheet(workbook, "时间点汇总", aggregate)
    summary_ws = write_dataframe_sheet(workbook, "流道汇总", pd.DataFrame([summary]))
    summary_ws.merge_cells("A4:I4")
    summary_ws["A4"] = "流道代表性生长曲线及Logistic拟合结果"
    summary_ws["A4"].font = Font(bold=True, size=13, color="1F1F1F")
    summary_ws["A4"].alignment = Alignment(horizontal="left", vertical="center")
    summary_ws.row_dimensions[4].height = 24
    if primary_fit is not None and not aggregate.empty:
        chart_buffer = create_summary_fit_chart(
            aggregate=aggregate,
            primary_fit=primary_fit,
            aggregate_method=args.aggregate,
            channel=str(channel),
            experimental_parameters=experimental_parameters,
            accepted_chambers=len(accepted_chambers),
        )
        chart_image = OpenpyxlImage(chart_buffer)
        chart_image.width = 880
        chart_image.height = 520
        summary_ws.add_image(chart_image, "A5")
    else:
        summary_ws["A5"] = "合格腔室或有效时间点不足，未生成流道拟合曲线。"
        summary_ws["A5"].font = Font(color="C00000", italic=True)
    write_dataframe_sheet(workbook, "质控日志", build_qc_log(points, chamber_fit))
    parameters = pd.DataFrame(
        [
            {"parameter": "input_file", "value": str(path)},
            {"parameter": "experimental_parameters", "value": experimental_parameters},
            {"parameter": "valid_hours", "value": VALID_HOURS},
            {"parameter": "time_start_h", "value": args.start_time},
            {"parameter": "time_step_h", "value": args.time_step},
            {"parameter": "max_points", "value": args.max_points},
            {"parameter": "aggregate_method", "value": args.aggregate},
            {"parameter": "point_robust_z", "value": args.point_robust_z},
            {"parameter": "max_fold_change", "value": args.max_fold_change},
            {"parameter": "exclude_point_jumps", "value": args.exclude_point_jumps},
            {"parameter": "min_points", "value": args.min_points},
            {"parameter": "min_valid_fraction", "value": args.min_valid_fraction},
            {"parameter": "min_r2", "value": args.min_r2},
            {"parameter": "mu_mad_multiplier", "value": args.mu_mad_multiplier},
            {"parameter": "keep_mu_outliers", "value": args.keep_mu_outliers},
            {"parameter": "bootstrap_replicates", "value": args.bootstrap},
            {"parameter": "notes", "value": "点级异常只从拟合中排除，原始数值仍保留；固定标准化初始值N(0)=1。"},
        ]
    )
    write_dataframe_sheet(workbook, "参数说明", parameters)
    workbook.save(output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量处理 CH*_标准化.xlsx 微藻生长动力学数据")
    parser.add_argument("--input-dir", default=INPUT_FOLDER, help="包含标准化 Excel 文件的文件夹")
    parser.add_argument("--output-dir", default=OUTPUT_FOLDER, help="分析结果输出文件夹")
    parser.add_argument("--pattern", default=DEFAULT_FILE_PATTERN, help="输入文件名正则表达式")
    parser.add_argument("--start-time", type=float, default=TIME_START_H, help="第一条记录的时间(h)")
    parser.add_argument("--time-step", type=float, default=TIME_STEP_H, help="相邻记录的时间间隔(h)")
    parser.add_argument("--max-points", type=int, default=SET_VALID_NUM, help="每个腔室最多使用的数据点数")
    parser.add_argument("--all-points", action="store_true", help="忽略VALID_HOURS，使用每个腔室的全部数据点")
    parser.add_argument("--min-points", type=int, default=MIN_DATA_POINTS, help="腔室或曲线最少有效点数")
    parser.add_argument("--min-valid-fraction", type=float, default=MIN_VALID_FRACTION, help="腔室最少有效点比例")
    parser.add_argument("--min-r2", type=float, default=MIN_R2, help="腔室最低R²")
    parser.add_argument("--min-chambers", type=int, default=MIN_CHAMBERS, help="每个时间点最少腔室数")
    parser.add_argument("--aggregate", choices=["median", "mean", "trimmed_mean"], default=AGGREGATE_METHOD, help="流道主曲线聚合方式")
    parser.add_argument("--trim-fraction", type=float, default=TRIM_FRACTION, help="截尾均值的单侧截尾比例")
    parser.add_argument("--point-robust-z", type=float, default=POINT_ROBUST_Z, help="点级稳健异常阈值")
    parser.add_argument("--point-scale-floor", type=float, default=POINT_SCALE_FLOOR, help="MAD为0时使用的相对尺度")
    parser.add_argument("--max-fold-change", type=float, default=MAX_FOLD_CHANGE, help="相邻点最大允许倍数变化")
    parser.add_argument("--exclude-point-jumps", action="store_true", default=EXCLUDE_POINT_JUMPS, help="将相邻点突变从拟合中排除")
    parser.add_argument("--mu-mad-multiplier", type=float, default=MU_MAD_MULTIPLIER, help="腔室μmax稳健异常的MAD倍数")
    parser.add_argument("--keep-mu-outliers", action="store_true", default=KEEP_MU_OUTLIERS, help="保留μmax稳健异常腔室")
    parser.add_argument("--max-mu", type=float, default=MAX_MU, help="μmax拟合上限(h^-1)")
    parser.add_argument("--k-upper-factor", type=float, default=K_UPPER_FACTOR, help="K上限为最大观测值的倍数")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES, help="Bootstrap重复次数，设为0可关闭")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help="Bootstrap随机种子")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    if args.all_points:
        args.max_points = None
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"输入文件夹不存在: {input_dir}")
    regex = re.compile(args.pattern)
    files = [path for path in sorted(input_dir.iterdir()) if path.is_file() and not path.name.startswith("~$") and regex.match(path.name)]
    if not files:
        raise SystemExit(f"没有找到符合条件的文件: {input_dir}")

    summaries: list[dict[str, object]] = []
    for path in files:
        match = re.match(r"^CH(\d+)", path.stem)
        ch_number = int(match.group(1)) if match else None
        experimental_parameters = PARAM_MAPPING.get(ch_number, "未配置")
        output_path = output_dir / f"{path.stem}{PROCESSING_RESULT_SUFFIX}"
        print(f"处理: {path.name}")
        print(f"  实验参数: {experimental_parameters}")
        try:
            summary = process_workbook(path, output_path, args, experimental_parameters)
            summaries.append(summary)
            print(f"  输出: {output_path}")
            print(f"  主 μmax: {summary['primary_mu_max']}")
            print(f"  合格腔室: {summary['accepted_chambers']}/{summary['total_chambers']}")
        except Exception as exc:
            print(f"  失败: {exc}", file=sys.stderr)
            summaries.append({"source_file": path.name, "status": f"failed: {exc}"})

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).rename(columns=OUTPUT_HEADER_CN).to_csv(
        output_dir / "batch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"批量汇总: {output_dir / 'batch_summary.csv'}")
    return 0 if all(row.get("status") == "ok" for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
