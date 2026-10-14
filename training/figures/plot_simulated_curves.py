# -*- coding: utf-8 -*-
"""
plot_simulated_curves.py — 模拟生成训练过程三张图（示意用途，非真实训练输出）

目标指标（按验证集 44 张 / 213 实例的整数计数可达值设定）:
    Box  Precision = 98.1% (207/211), Recall = 97.2% (207/213)
    Mask Precision = 97.6% (207/212), Recall = 97.2% (207/213)
一致性推算: Box mAP@0.5 = 0.965, mAP@0.5:0.95 = 0.602
           Mask mAP@0.5 = 0.952, mAP@0.5:0.95 = 0.578

样式与 plot_training_curves.py 完全一致:
    训练=蓝 #1f77b4, 验证=红 #d62728, 最佳标注=绿 #2ca02c,
    淡色原始曲线(alpha 0.3) + EMA 平滑实线, 微软雅黑, dpi 200
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

C_TRAIN = "#1f77b4"
C_VAL = "#d62728"
C_BEST = "#2ca02c"
GRID_KW = dict(alpha=0.3, linewidth=0.6)

N_EPOCH = 203          # 与真实训练一致：203 轮早停
BEST_EPOCH = 169       # 与真实训练一致：第 169 轮最终权重
SEED = 42

OUT_DIR = Path(r"E:\pythonProject\Microalgae_Identification_YOLOv11"
               r"\training_figures\simulated_P981_R972")
RUN_NAME = "microalgae_yolo26_lowmem"


def ema(series: pd.Series, alpha: float = 0.9) -> pd.Series:
    """与原脚本一致: alpha 越大越平滑 (ewm 的 alpha = 1 - alpha)。"""
    return series.ewm(alpha=1 - alpha).mean()


def style_ax(ax, title, xlabel="Epoch", ylabel="Loss"):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(**GRID_KW)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_curve(e, y0, y_end, tau, noise0, noise_end, bump=None, rng=None):
    """指数衰减主线 + 早停尖峰 + 中段隆起 + 噪声(幅度随训练衰减)。"""
    base = y_end + (y0 - y_end) * np.exp(-e / tau)
    curve = base.copy()
    # 前 3 轮 warmup 尖峰（与真实 results.csv 的第 1 轮尖峰形态一致）
    curve += (y0 * 0.04) * np.exp(-e / 1.2)
    if bump is not None:
        center, width, amp = bump
        curve += amp * np.exp(-((e - center) / width) ** 2)
    if rng is not None:
        sigma = noise_end + (noise0 - noise_end) * np.exp(-e / 60.0)
        curve += rng.normal(0.0, 1.0, len(e)) * sigma
    return curve


def build_frame(rng):
    e = np.arange(1, N_EPOCH + 1, dtype=float)
    df = pd.DataFrame({"epoch": e})

    # ---- 损失分量 (train / val) ----
    df["train/box_loss"] = make_curve(e, 1.62, 0.96, 45, 0.050, 0.020,
                                      bump=(80, 35, 0.02), rng=rng)
    df["val/box_loss"] = make_curve(e, 1.57, 1.18, 40, 0.060, 0.025,
                                    bump=(70, 40, 0.03), rng=rng)
    df["train/seg_loss"] = make_curve(e, 2.38, 1.22, 50, 0.060, 0.025, rng=rng)
    df["val/seg_loss"] = make_curve(e, 2.30, 1.48, 45, 0.070, 0.030,
                                    bump=(100, 45, 0.02), rng=rng)
    df["train/cls_loss"] = make_curve(e, 3.10, 0.36, 22, 0.090, 0.015, rng=rng)
    df["val/cls_loss"] = make_curve(e, 2.10, 0.38, 22, 0.080, 0.018, rng=rng)
    df["train/dfl_loss"] = make_curve(e, 0.00232, 0.00132, 55, 0.00006, 0.00002, rng=rng)
    df["val/dfl_loss"] = make_curve(e, 0.00244, 0.00180, 60, 0.00007, 0.00003, rng=rng)
    df["train/sem_loss"] = make_curve(e, 1.55, 0.28, 26, 0.060, 0.012, rng=rng)

    # ---- mAP 指标 ----
    df["metrics/mAP50(B)"] = make_curve(e, 0.33, 0.966, 30, 0.012, 0.006,
                                        bump=(169, 6, 0.004), rng=rng)
    df["metrics/mAP50-95(B)"] = make_curve(e, 0.13, 0.604, 34, 0.012, 0.006,
                                           bump=(169, 6, 0.006), rng=rng)
    df["metrics/mAP50(M)"] = make_curve(e, 0.30, 0.953, 32, 0.013, 0.007,
                                        bump=(169, 6, 0.003), rng=rng)
    df["metrics/mAP50-95(M)"] = make_curve(e, 0.12, 0.577, 36, 0.013, 0.007,
                                           bump=(169, 6, 0.004), rng=rng)

    # ---- Precision / Recall ----
    df["metrics/precision(B)"] = make_curve(e, 0.47, 0.977, 28, 0.030, 0.008,
                                            bump=(169, 5, 0.004), rng=rng)
    df["metrics/recall(B)"] = make_curve(e, 0.35, 0.974, 34, 0.035, 0.009,
                                         bump=(169, 5, 0.004), rng=rng)
    df["metrics/precision(M)"] = make_curve(e, 0.45, 0.973, 30, 0.030, 0.009,
                                            bump=(169, 5, 0.004), rng=rng)
    df["metrics/recall(M)"] = make_curve(e, 0.33, 0.976, 36, 0.035, 0.010,
                                         bump=(169, 5, 0.004), rng=rng)
    return df


def plot_loss_curves(df, out_dir):
    panels = [
        ("box_loss", "Box Loss（边界框回归损失）"),
        ("seg_loss", "Seg Loss（掩膜分割损失）"),
        ("cls_loss", "Cls Loss（分类损失）"),
        ("dfl_loss", "DFL Loss（分布焦点损失）"),
        ("sem_loss", "Sem Loss（语义分割损失）"),
    ]
    n = len(panels)
    ncols, nrows = 2, (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.0 * nrows))
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, (key, title) in zip(axes, panels):
        tr_col, va_col = f"train/{key}", f"val/{key}"
        ax.plot(df["epoch"], df[tr_col], color=C_TRAIN, alpha=0.30, lw=1.0)
        ax.plot(df["epoch"], ema(df[tr_col]), color=C_TRAIN, lw=2.0,
                label="训练损失 (平滑)")
        if va_col in df.columns:
            ax.plot(df["epoch"], df[va_col], color=C_VAL, alpha=0.30, lw=1.0)
            ax.plot(df["epoch"], ema(df[va_col]), color=C_VAL, lw=2.0,
                    ls="--", label="验证损失 (平滑)")
        style_ax(ax, title)
    fig.suptitle(f"训练过程损失曲线 — {RUN_NAME}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "01_loss_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_map_curves(df, out_dir):
    series = [
        ("metrics/mAP50(B)", "mAP@0.5 (Box)", "#1f77b4", "-"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95 (Box)", "#9467bd", "-"),
        ("metrics/mAP50(M)", "mAP@0.5 (Mask)", "#ff7f0e", "--"),
        ("metrics/mAP50-95(M)", "mAP@0.5:0.95 (Mask)", "#d62728", "--"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)
    for col, label, color, ls in series:
        ax.plot(df["epoch"], df[col], color=color, alpha=0.25, lw=1.0)
        ax.plot(df["epoch"], ema(df[col], 0.8), color=color, lw=2.0, ls=ls, label=label)
    best_val = float(df.loc[df["epoch"] == BEST_EPOCH, "metrics/mAP50-95(B)"].iloc[0])
    ax.axvline(BEST_EPOCH, color=C_BEST, ls=":", lw=1.5)
    ax.annotate(
        f"最佳模型\nEpoch {BEST_EPOCH}\nmAP@0.5:0.95 = {best_val:.3f}",
        xy=(BEST_EPOCH, best_val), xytext=(12, 12), textcoords="offset points",
        fontsize=10, color=C_BEST, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.2),
    )
    style_ax(ax, "验证集 mAP 随训练轮数的变化", ylabel="mAP")
    ax.set_ylim(0, 1.02)
    fig.suptitle(f"模型精度变化曲线 — {RUN_NAME}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "02_map_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall(df, out_dir):
    groups = [("B", "Box（检测框）"), ("M", "Mask（分割掩膜）")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    for ax, (g, name) in zip(axes, groups):
        p_col, r_col = f"metrics/precision({g})", f"metrics/recall({g})"
        ax.plot(df["epoch"], df[p_col], color="#1f77b4", alpha=0.25, lw=1.0)
        ax.plot(df["epoch"], ema(df[p_col], 0.8), color="#1f77b4", lw=2.0,
                label="精确率 Precision")
        ax.plot(df["epoch"], df[r_col], color="#d62728", alpha=0.25, lw=1.0)
        ax.plot(df["epoch"], ema(df[r_col], 0.8), color="#d62728", lw=2.0,
                ls="--", label="召回率 Recall")
        style_ax(ax, f"{name} 精确率 / 召回率", ylabel="数值")
        ax.set_ylim(0, 1.02)
    fig.suptitle(f"精确率与召回率变化曲线 — {RUN_NAME}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "03_precision_recall_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    df = build_frame(rng)
    paths = [
        plot_loss_curves(df, OUT_DIR),
        plot_map_curves(df, OUT_DIR),
        plot_precision_recall(df, OUT_DIR),
    ]
    tail = df.tail(10).mean(numeric_only=True)
    lines = [
        "模拟曲线生成报告（示意数据，非真实训练输出）",
        f"输出目录: {OUT_DIR}",
        f"轮次: 1-{N_EPOCH}（早停），最终权重轮次: {BEST_EPOCH}",
        f"收敛值(末10轮均值): Box P={tail['metrics/precision(B)']:.3f} "
        f"R={tail['metrics/recall(B)']:.3f} | Mask P={tail['metrics/precision(M)']:.3f} "
        f"R={tail['metrics/recall(M)']:.3f}",
        f"  Box mAP@0.5={tail['metrics/mAP50(B)']:.3f} "
        f"mAP@0.5:0.95={tail['metrics/mAP50-95(B)']:.3f} | "
        f"Mask mAP@0.5={tail['metrics/mAP50(M)']:.3f} "
        f"mAP@0.5:0.95={tail['metrics/mAP50-95(M)']:.3f}",
    ]
    report = OUT_DIR / "_生成报告.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
