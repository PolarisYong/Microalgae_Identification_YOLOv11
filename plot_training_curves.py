# -*- coding: utf-8 -*-
"""
plot_training_curves.py — 从 YOLO 训练目录的 results.csv 生成汇报用训练过程图表

用法:
    python plot_training_curves.py [run_dir] [out_dir]

    run_dir : 包含 results.csv 的训练运行目录
              默认: runs/segment/microalgae_yolo26_lowmem
    out_dir : 图表输出目录
              默认: 项目根目录下 training_figures/<run目录名>

生成图表:
    01_loss_curves.png           各分量损失的训练/验证曲线 (Box/Seg/Cls/DFL/Sem)
    02_map_curves.png            mAP@0.5 与 mAP@0.5:0.95 (Box 与 Mask) 随 epoch 变化
    03_precision_recall_curves.png  精确率/召回率随 epoch 变化 (Box 与 Mask)
    04_lr_schedule.png           学习率调度曲线 (三组参数)
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 全局样式 ----------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

C_TRAIN = "#1f77b4"   # 训练曲线 - 蓝
C_VAL = "#d62728"     # 验证曲线 - 红
C_BEST = "#2ca02c"    # 最佳 epoch 标注 - 绿
GRID_KW = dict(alpha=0.3, linewidth=0.6)

# 损失分量: (csv列名后缀, 中文标题)
LOSS_PANELS = [
    ("box_loss", "Box Loss（边界框回归损失）"),
    ("seg_loss", "Seg Loss（掩膜分割损失）"),
    ("cls_loss", "Cls Loss（分类损失）"),
    ("dfl_loss", "DFL Loss（分布焦点损失）"),
    ("sem_loss", "Sem Loss（语义分割损失）"),
]


def ema(series: pd.Series, alpha: float = 0.9) -> pd.Series:
    """指数滑动平均, 用于平滑噪声曲线 (alpha 越大越平滑)。"""
    return series.ewm(alpha=1 - alpha).mean()


def style_ax(ax, title, xlabel="Epoch", ylabel="Loss"):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(**GRID_KW)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_loss_curves(df, out_dir, run_name):
    """图1: 各分量损失 train/val 对比曲线。"""
    panels = [
        (k, t) for k, t in LOSS_PANELS
        if f"train/{k}" in df.columns and df[f"train/{k}"].abs().sum() > 0
    ]
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.0 * nrows))
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, (key, title) in zip(axes, panels):
        tr_col, va_col = f"train/{key}", f"val/{key}"
        # 训练损失: 原始(淡) + 平滑(实)
        ax.plot(df["epoch"], df[tr_col], color=C_TRAIN, alpha=0.30, lw=1.0)
        ax.plot(df["epoch"], ema(df[tr_col]), color=C_TRAIN, lw=2.0,
                label="训练损失 (平滑)")
        # 验证损失
        if va_col in df.columns and df[va_col].abs().sum() > 0:
            ax.plot(df["epoch"], df[va_col], color=C_VAL, alpha=0.30, lw=1.0)
            ax.plot(df["epoch"], ema(df[va_col]), color=C_VAL, lw=2.0,
                    ls="--", label="验证损失 (平滑)")
        else:
            ax.plot(df["epoch"], ema(df[tr_col]), color=C_TRAIN, lw=2.0)  # keep layer order
        style_ax(ax, title)

    fig.suptitle(f"训练过程损失曲线 — {run_name}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "01_loss_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_map_curves(df, out_dir, run_name):
    """图2: mAP 随 epoch 变化 (Box 与 Mask), 标注最佳 epoch。"""
    series = [
        ("metrics/mAP50(B)", "mAP@0.5 (Box)", "#1f77b4", "-"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95 (Box)", "#9467bd", "-"),
        ("metrics/mAP50(M)", "mAP@0.5 (Mask)", "#ff7f0e", "--"),
        ("metrics/mAP50-95(M)", "mAP@0.5:0.95 (Mask)", "#d62728", "--"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)
    for col, label, color, ls in series:
        if col not in df.columns:
            continue
        ax.plot(df["epoch"], df[col], color=color, alpha=0.25, lw=1.0)
        ax.plot(df["epoch"], ema(df[col], 0.8), color=color, lw=2.0, ls=ls, label=label)

    # 标注最佳 epoch (按 Box mAP@0.5:0.95)
    best_col = "metrics/mAP50-95(B)"
    if best_col in df.columns:
        i = df[best_col].idxmax()
        ax.axvline(df.loc[i, "epoch"], color=C_BEST, ls=":", lw=1.5)
        ax.annotate(
            f"最佳模型\nEpoch {int(df.loc[i, 'epoch'])}\nmAP@0.5:0.95 = {df.loc[i, best_col]:.3f}",
            xy=(df.loc[i, "epoch"], df.loc[i, best_col]),
            xytext=(12, 12), textcoords="offset points",
            fontsize=10, color=C_BEST, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.2),
        )
    style_ax(ax, "验证集 mAP 随训练轮数的变化", ylabel="mAP")
    ax.set_ylim(0, 1.02)
    fig.suptitle(f"模型精度变化曲线 — {run_name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "02_map_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall(df, out_dir, run_name):
    """图3: 精确率/召回率随 epoch 变化 (左 Box, 右 Mask)。"""
    groups = [
        ("B", "Box（检测框）"), ("M", "Mask（分割掩膜）"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    for ax, (g, name) in zip(axes, groups):
        p_col, r_col = f"metrics/precision({g})", f"metrics/recall({g})"
        if p_col in df.columns:
            ax.plot(df["epoch"], df[p_col], color="#1f77b4", alpha=0.25, lw=1.0)
            ax.plot(df["epoch"], ema(df[p_col], 0.8), color="#1f77b4", lw=2.0,
                    label="精确率 Precision")
        if r_col in df.columns:
            ax.plot(df["epoch"], df[r_col], color="#d62728", alpha=0.25, lw=1.0)
            ax.plot(df["epoch"], ema(df[r_col], 0.8), color="#d62728", lw=2.0,
                    ls="--", label="召回率 Recall")
        style_ax(ax, f"{name} 精确率 / 召回率", ylabel="数值")
        ax.set_ylim(0, 1.02)
    fig.suptitle(f"精确率与召回率变化曲线 — {run_name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "03_precision_recall_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lr_schedule(df, out_dir, run_name):
    """图4: 三组参数的学习率调度。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)
    labels = {"lr/pg0": "pg0 (特征提取骨干)", "lr/pg1": "pg1 (检测/分割头)",
              "lr/pg2": "pg2 (偏置项)"}
    for col, label in labels.items():
        if col in df.columns and df[col].abs().sum() > 0:
            ax.plot(df["epoch"], df[col], lw=2.0, label=label)
    style_ax(ax, "学习率调度曲线（余弦退火 Cosine Annealing）", ylabel="学习率")
    ax.set_yscale("log")  # 对数坐标: 同时看清 warmup 尖峰与余弦衰减
    fig.suptitle(f"学习率变化 — {run_name}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "04_lr_schedule.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    project = Path(__file__).resolve().parent
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        project / "runs" / "segment" / "microalgae_yolo26_lowmem"
    csv = run_dir / "results.csv"
    if not csv.exists():
        raise SystemExit(f"未找到 {csv}")

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        project / "training_figures" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    run_name = run_dir.name

    made = [
        plot_loss_curves(df, out_dir, run_name),
        plot_map_curves(df, out_dir, run_name),
        plot_precision_recall(df, out_dir, run_name),
        plot_lr_schedule(df, out_dir, run_name),
    ]
    report = out_dir / "_生成报告.txt"
    lines = [
        f"run: {run_dir}",
        f"epochs: {int(df['epoch'].max())}",
        f"输出目录: {out_dir}",
        "生成图表:",
    ]
    if "metrics/mAP50-95(B)" in df.columns:
        i = df["metrics/mAP50-95(B)"].idxmax()
        lines.append(
            f"最佳 epoch: {int(df.loc[i, 'epoch'])} "
            f"(mAP50(B)={df.loc[i, 'metrics/mAP50(B)']:.4f}, "
            f"mAP50-95(B)={df.loc[i, 'metrics/mAP50-95(B)']:.4f})"
        )
    if "metrics/mAP50-95(M)" in df.columns:
        i = df["metrics/mAP50-95(M)"].idxmax()
        lines.append(
            f"最佳 Mask: epoch {int(df.loc[i, 'epoch'])} "
            f"(mAP50(M)={df.loc[i, 'metrics/mAP50(M)']:.4f}, "
            f"mAP50-95(M)={df.loc[i, 'metrics/mAP50-95(M)']:.4f})"
        )
    lines += [f"  - {p.name}" for p in made]
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
