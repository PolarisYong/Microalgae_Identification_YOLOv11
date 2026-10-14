# -*- coding: utf-8 -*-
"""
英文版图表（Y = ln(Nrel) 尺度）
================================
按「图表文字内容修改意见.pptx」重绘英文版图：

1. **全部生长曲线画在 Y = ln(Nrel) 尺度**：散点取 ln(实测相对数量)，
   曲线为 Y(t) = lnKrel / (1 + exp(4μmax/lnKrel·(λ−t) + 2))（修改意见第 1 页公式），
   散点与曲线严格同尺度，平台 ≈ ln Krel。
2. 参数框：λ 带单位 h；拟合优度写作 R²_Y（其余与原来一致，
   Krel 仍报告原值，不取对数——按修改意见明确要求）。
3. 文字规范：
   - 合并生长图  Growth curve (merged data, N chambers, L30-N20-DIC75)
                 y = Log relative cell number, Y
   - 合并面积图  Atot,rel / Amean,rel (merged data, N chambers, <cond>)
                 y = Relative total / mean single-cell cell projected area, Atot,rel / Amean,rel
   - 腔室图      Growth curve (CB-1, <cond>) / CB-1 data vs merged fitted curve (<cond>)
                 Atot,rel (CB-1, <cond>) / Amean,rel (CB-1, <cond>)
   - 横轴统一    Cultivation time, t (h)
4. 不重新拟合：曲线全部由工作簿里已存的 (μmax, Krel, λ) 参数解析生成。
"""
import io

import numpy as np

import config
from core.fitting import fit_model
from core.plotting import prepare, clean_label

X_LABEL = "Cultivation time, $t$ (h)"
Y_LABEL_GROWTH = "Log relative cell number, $Y$"
# 下标格式（用户要求）：tot,rel / mean,rel 用 mathtext 下标
ATOT_MATH = r"$A_{\mathrm{tot,rel}}$"
AMEAN_MATH = r"$A_{\mathrm{mean,rel}}$"
Y_LABEL_ATOT = f"Relative total cell projected area, {ATOT_MATH}"
Y_LABEL_AMEAN = f"Relative mean single-cell projected area, {AMEAN_MATH}"


# --------------------------------------------------------------------------- 基础
def _finish(fig, dpi=None):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi or config.PLOT_DPI, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _style_time_axis(ax):
    """与 core.plotting 一致的时间轴刻度，横轴标签按修改意见写法。"""
    from matplotlib.ticker import MultipleLocator

    ax.set_xlabel(X_LABEL)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(3))
    ticks = ax.get_yticks()
    interval = (ticks[1] - ticks[0]) if len(ticks) > 1 else 1
    if interval == 0:
        interval = 1
    ax.yaxis.set_minor_locator(MultipleLocator(interval / 5))


def _param_text(mu, krel, n0, lam, r2):
    """图内参数框：λ 补单位 h、R² 写作 R²_Y（修改意见第 1 页批注）。"""
    return (
        f"$\\it{{\\mu}}_{{\\mathrm{{max}}}}$ = {mu:.4f} h⁻¹\n"
        f"$K_{{\\mathrm{{rel}}}}$ = {krel:.2f} (dimensionless)\n"
        f"$N_{{\\mathrm{{rel,fit}}}}(0)$ = {n0:.2f}\n"
        f"$\\it{{\\lambda}}$ = {lam:.2f} h\n"
        f"$R^{{2}}_{{Y}}$ = {r2:.4f}"
    )


def _param_box(ax, text):
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _y_curve(t, mu, krel, lam):
    """Y(t) = lnKrel / (1 + exp(4μmax/lnKrel·(λ−t) + 2))，修改意见第 1 页公式。"""
    return fit_model(np.asarray(t, dtype=float), float(mu),
                     np.log(float(krel)), float(lam))


def _ln(values):
    arr = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(arr)


def _new_fig(figsize=(10, 6)):
    import matplotlib.pyplot as plt

    prepare()
    return plt.figure(figsize=figsize)


# --------------------------------------------------------------------------- 腔室页 4 张图
def chamber_fit_figure_en(t, counts, params, cb_id, condition):
    """单腔室生长曲线拟合（Y 尺度）：蓝点 ln(实测) + 红拟合曲线 + 参数框。"""
    import matplotlib.pyplot as plt

    fig = _new_fig()
    plt.scatter(t, _ln(counts), label="Actual data", color="blue", alpha=0.6)
    plt.plot(t, _y_curve(t, params["mu"], params["K"], params["lam"]),
             label="Fitted curve", color="red", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_GROWTH)
    ax.set_title(f"Growth curve ({cb_id}, {clean_label(condition)})")
    _style_time_axis(ax)
    _param_box(ax, _param_text(params["mu"], params["K"], params["N0"],
                               params["lam"], params["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def chamber_vs_merged_figure_en(t, counts, merged, cb_id, condition):
    """该腔室 ln(实测) + 全组合并拟合曲线（黄，Y 尺度）。"""
    import matplotlib.pyplot as plt

    fig = _new_fig()
    plt.scatter(t, _ln(counts), label="Actual data", color="blue", alpha=0.6)
    plt.plot(t, _y_curve(t, merged["mu"], merged["K"], merged["lam"]),
             label="Merged fitted curve", color="#F2BA02", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_GROWTH)
    ax.set_title(f"{cb_id} data vs merged fitted curve ({clean_label(condition)})")
    _style_time_axis(ax)
    _param_box(ax, _param_text(merged["mu"], merged["K"], merged["N0"],
                               merged["lam"], merged["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def chamber_area_figures_en(t, area, avg_area, cb_id, condition):
    """Atot,rel（绿散点）+ Amean,rel（紫散点+连线）。"""
    import matplotlib.pyplot as plt

    cond = clean_label(condition)
    fig = _new_fig()
    plt.scatter(t, area, label=Y_LABEL_ATOT, color="green", alpha=0.6)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_ATOT)
    ax.set_title(f"{ATOT_MATH} ({cb_id}, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    total_buf = _finish(fig)

    fig = _new_fig()
    plt.scatter(t, avg_area, label=Y_LABEL_AMEAN, color="purple", alpha=0.6, s=30)
    plt.plot(t, avg_area, color="purple", alpha=0.8, linestyle="-", linewidth=1.5)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_AMEAN)
    ax.set_title(f"{AMEAN_MATH} ({cb_id}, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    avg_buf = _finish(fig)
    return total_buf, avg_buf


# --------------------------------------------------------------------------- 汇总页 3 张图
def merged_growth_figure_en(merged_t, merged_counts, merged, n_chambers, condition):
    """全部腔室 ln(实测) + 合并拟合曲线（Y 尺度）。"""
    import matplotlib.pyplot as plt

    cond = clean_label(condition)
    fig = _new_fig(figsize=(12, 7))
    plt.scatter(merged_t, _ln(merged_counts), label="Merged actual data",
                color="blue", alpha=0.5, s=30)
    curve_t = np.unique(np.asarray(merged_t, dtype=float))
    plt.plot(curve_t, _y_curve(curve_t, merged["mu"], merged["K"], merged["lam"]),
             label="Merged fitted curve", color="#F2BA02", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_GROWTH)
    ax.set_title(f"Growth curve (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    _param_box(ax, _param_text(merged["mu"], merged["K"], merged["N0"],
                               merged["lam"], merged["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def merged_area_figures_en(t_all, area_all, avg_all, n_chambers, condition):
    """全部腔室的 Atot,rel / Amean,rel 散点。"""
    import matplotlib.pyplot as plt

    cond = clean_label(condition)
    fig = _new_fig(figsize=(12, 7))
    plt.scatter(t_all, area_all, label="Relative total cell projected area",
                color="green", alpha=0.6, s=30)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_ATOT)
    ax.set_title(f"{ATOT_MATH} (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    total_buf = _finish(fig)

    fig = _new_fig(figsize=(12, 7))
    plt.scatter(t_all, avg_all, label="Relative mean single-cell projected area",
                color="purple", alpha=0.6, s=30)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_AMEAN)
    ax.set_title(f"{AMEAN_MATH} (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    avg_buf = _finish(fig)
    return total_buf, avg_buf


# --------------------------------------------------------------------------- PPT 腔室图
def ppt_chamber_figure_en(t, counts, params, cb_id, condition, size_px):
    """
    PPT 配图：与腔室页拟合图同款（Y 尺度），但按原图素尺寸渲染以保版式。
    size_px = (宽, 高)，原 PPT 图片为 300 dpi（如 3054×1983）。
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    width, height = size_px
    dpi = 300
    fig = _new_fig(figsize=(width / dpi, height / dpi))
    plt.scatter(t, _ln(counts), label="Actual data", color="blue", alpha=0.6, s=18)
    plt.plot(t, _y_curve(t, params["mu"], params["K"], params["lam"]),
             label="Fitted curve", color="red", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel(Y_LABEL_GROWTH)
    ax.set_title(f"Growth curve ({cb_id}, {clean_label(condition)})")
    _style_time_axis(ax)
    _param_box(ax, _param_text(params["mu"], params["K"], params["N0"],
                               params["lam"], params["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    buf = _finish(fig, dpi=dpi)
    # bbox_inches="tight" 会改变像素尺寸，缩放到与原图一致以保版式
    img = Image.open(buf)
    if img.size != (width, height):
        out = io.BytesIO()
        img.resize((width, height), Image.LANCZOS).save(out, format="png")
        out.seek(0)
        return out
    buf.seek(0)
    return buf
