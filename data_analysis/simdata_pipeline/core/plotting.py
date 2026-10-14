# -*- coding: utf-8 -*-
"""
可视化图表
==========
移植自参考实现 `08_processed_batch.py` 的绘图部分。参考实现用 matplotlib 生成
PNG，再通过 `openpyxl` 的 `ws.add_image` 嵌入工作簿（不是 Excel 原生图表对象），
因此本模块只负责「生成 PNG + 插到指定单元格」。

嵌图位置与参考实现的实际输出一致：

    腔室页 CH1_CB<n>   4 张 —— H2  单腔室生长曲线拟合
                                H24 该腔室数据 vs 合并拟合曲线
                                R2  相对总细胞面积
                                R24 相对平均细胞面积
    汇总页              3 张 —— 与「合并拟合结果」同一行，从 C 列起向右排开
                                生长曲线（合并）/ 相对总面积（合并）/ 相对平均面积（合并）
                                （列间距由 writer._merged_anchors 按列宽计算，保证不重叠）

坐标轴、标题、图例文字均与参考实现逐字一致（英文），因此不存在中文字体问题。
"""
import io

import numpy as np

import config

_FONT_READY = False


# --------------------------------------------------------------------------- 基础
def prepare():
    """配置 matplotlib（只需一次）。rcParams 与参考实现逐项一致。"""
    global _FONT_READY
    if _FONT_READY:
        return
    import matplotlib
    matplotlib.use("Agg")           # 无窗口后端，服务器/后台运行安全
    from matplotlib import rcParams

    rcParams["font.family"] = config.PLOT_FONT_FAMILY
    rcParams["axes.unicode_minus"] = False
    # 数学字体：mathtext 默认 dejavusans 与 Times New Roman 不协调，
    # stix 是专为 Times 系正文设计的数学字体（μmax、Krel、A_{tot,rel} 等）
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 20
    rcParams["xtick.labelsize"] = 20
    rcParams["ytick.labelsize"] = 20
    rcParams["axes.titlesize"] = 24
    rcParams["legend.fontsize"] = 18
    rcParams["axes.titley"] = 1.01
    _FONT_READY = True


def clean_label(text):
    """
    目录名用的是 Unicode 连字符 U+2011（-），Times New Roman 无此字形，
    直接用会渲染成方框。绘图标题统一换成 ASCII '-'。
    """
    return str(text).replace("\u2011", "-").replace("\u2010", "-")


def _finish(fig):
    """fig → PNG 字节流（dpi / bbox 与参考实现一致）。"""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=config.PLOT_DPI, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _style_time_axis(ax):
    """时间轴：主刻度 12 h、次刻度 3 h；y 轴次刻度取主刻度的 1/5。"""
    from matplotlib.ticker import MultipleLocator

    ax.set_xlabel("Cultivation time (h)")
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(3))
    ticks = ax.get_yticks()
    interval = (ticks[1] - ticks[0]) if len(ticks) > 1 else 1
    if interval == 0:
        interval = 1
    ax.yaxis.set_minor_locator(MultipleLocator(interval / 5))


def _param_text(mu, krel, n0, lam, r2, tight=False):
    """
    图内参数文本框。`tight=True` 用参考实现单腔室版本的写法（`μmax`= 后紧跟空格），
    False 用合并拟合版本的写法（`μmax` = ...）。
    """
    eq = "=" if tight else " ="
    return (
        f"$\\it{{μ}}_{{\\mathrm{{max}}}}${eq} {mu:.4f} h⁻¹\n"
        f"$K_{{rel}}$ = {krel:.2f} (dimensionless)\n"
        f"$N_{{rel,fit}}(0)$ = {n0:.2f}\n"
        f"$\\it{{λ}}$ = {lam:.2f}\n"
        f"R²(Y) = {r2:.4f}"
    )


def _param_box(ax, text):
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=20,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def _screen(values):
    """过滤 NaN / inf，避免把无效点画进图里。"""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


# --------------------------------------------------------------------------- 腔室页 4 张图
def chamber_fit_figure(t, counts, fit, plt_name):
    """单腔室生长曲线拟合：蓝点实测 + 红拟合曲线 + 参数框。"""
    import matplotlib.pyplot as plt

    prepare()
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, counts, label="Actual data", color="blue", alpha=0.6)
    plt.plot(t, fit["y_fit"], label="Fitted curve", color="red", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel("Relative cell number (normalized to 0 h)")
    ax.set_title(f"{plt_name} Cell growth curve fitting")
    _style_time_axis(ax)
    _param_box(ax, _param_text(fit["mu_max"], fit["K"], fit["N0"],
                               fit["lambda_"], fit["r2"], tight=True))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def chamber_vs_merged_figure(t, counts, merged_fit, plt_name):
    """该腔室实测点 + 全组合并拟合曲线（黄色）。"""
    import matplotlib.pyplot as plt
    from core.fitting import model

    prepare()
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, counts, label="Actual data", color="blue", alpha=0.6)
    merged_curve = model(t, merged_fit["mu_fit"], merged_fit["K"], merged_fit["lambda_"])
    plt.plot(t, merged_curve, label="Merged fitted curve", color="#F2BA02", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel("Relative cell number (normalized to 0 h)")
    ax.set_title(f"{plt_name} data vs merged fitting curve")
    _style_time_axis(ax)
    _param_box(ax, _param_text(merged_fit["mu_max"], merged_fit["K"], merged_fit["N0"],
                               merged_fit["lambda_"], merged_fit["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def chamber_area_figures(t, area, avg_area, plt_name):
    """相对总细胞面积（绿）+ 相对平均细胞面积（紫）两张图。"""
    import matplotlib.pyplot as plt

    prepare()
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, area, label="Relative total cell area", color="green", alpha=0.6)
    ax = plt.gca()
    ax.set_ylabel("Relative total cell area (normalized to 0 h)")
    ax.set_title(f"{plt_name} - Relative total cell area variation")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    total_buf = _finish(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(t, avg_area, label="Relative average cell area",
                color="purple", alpha=0.6, s=30)
    plt.plot(t, avg_area, color="purple", alpha=0.8, linestyle="-",
             linewidth=1.5, marker="", markersize=5)
    ax = plt.gca()
    ax.set_ylabel("Relative average cell area (normalized to 0 h)")
    ax.set_title(f"{plt_name} - Relative average cell area variation")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    avg_buf = _finish(fig)

    return total_buf, avg_buf


# --------------------------------------------------------------------------- 汇总页 3 张图
def merged_growth_figure(merged_t, merged_counts, merged_fit, n_chambers, condition):
    """全部腔室实测点 + 合并拟合曲线。"""
    import matplotlib.pyplot as plt
    from core.fitting import model

    prepare()
    cond = clean_label(condition)
    fig = plt.figure(figsize=(12, 7))
    plt.scatter(merged_t, merged_counts, label="Merged actual data",
                color="blue", alpha=0.5, s=30)
    curve_t = np.unique(np.asarray(merged_t, dtype=float))
    plt.plot(curve_t,
             model(curve_t, merged_fit["mu_fit"], merged_fit["K"], merged_fit["lambda_"]),
             label="Merged fitted curve", color="#F2BA02", linewidth=2)
    ax = plt.gca()
    ax.set_ylabel("Relative cell number (normalized to 0 h)")
    ax.set_title(f"Growth curve (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    _param_box(ax, _param_text(merged_fit["mu_max"], merged_fit["K"], merged_fit["N0"],
                               merged_fit["lambda_"], merged_fit["r2"]))
    plt.legend()
    plt.grid(alpha=0.3)
    return _finish(fig)


def merged_area_figures(t_all, area_all, avg_all, n_chambers, condition):
    """全部腔室的相对总面积 / 相对平均面积散点（同一张图叠所有腔室）。"""
    import matplotlib.pyplot as plt

    prepare()
    cond = clean_label(condition)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(t_all, area_all, label="Relative total area data",
                color="green", alpha=0.6, s=30)
    ax = plt.gca()
    ax.set_ylabel("Relative total cell area (normalized to 0 h)")
    ax.set_title(f"Relative total area (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    total_buf = _finish(fig)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(t_all, avg_all, label="Relative average cell area data",
                color="purple", alpha=0.6, s=30)
    ax = plt.gca()
    ax.set_ylabel("Relative average cell area (normalized to 0 h)")
    ax.set_title(f"Relative average cell area (merged data, {n_chambers} chambers, {cond})")
    _style_time_axis(ax)
    plt.legend()
    plt.grid(alpha=0.3)
    avg_buf = _finish(fig)

    return total_buf, avg_buf


# --------------------------------------------------------------------------- 写入
def insert_image(ws, buf, anchor, size):
    """把 PNG 字节流插到页签的指定锚点，并按 `size=(宽, 高)` 像素设定显示尺寸。"""
    from openpyxl.drawing.image import Image

    img = Image(buf)
    img.width, img.height = size
    ws.add_image(img, anchor)
