# -*- coding: utf-8 -*-
"""plot_s3_revised.py
按导师批注重绘 S3 全部插图（英文轴标题 / Times New Roman / 2x2 组图 / 子图仅保留序号）。

输出目录: training_figures/s3_revised/
    figS3-1_labels_distribution.png   训练集标注分布（2x2）
    figS3-2_training_performance.png  训练损失与验证性能（2x2）
    figS3-3_roi_overlay.png           ROI 边界提取（1x3）
    figS3-4_precision_recall.png      精确率/召回率曲线（1x2）
    figS3-5_gt_vs_pred.png            真值 vs 预测对照（1x2）
"""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
OUT = os.path.join(ROOT, "training_figures", "s3_revised")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

# ---------------- 全局样式：Times New Roman，轴标题 24pt 加粗，刻度 20pt ----------------
TNR = "Times New Roman"
assert any("Times New Roman" in f.name for f in font_manager.fontManager.ttflist), "Times New Roman 不可用"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [TNR, "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "figure.dpi": 300,
})
BLUE, RED, GREEN, GRAY, PURPLE, ORANGE = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd", "#ff7f0e"
GRID = dict(alpha=0.30, linewidth=0.8, which="major")
PANEL_FS = 24


def panel_label(ax, s, dx=-0.14, dy=1.04):
    """子图序号 (a)~(d)：新罗马、加粗、置于子图左上角，仅保留序号。"""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight="bold", va="bottom", ha="left", family=TNR)


def epoch_marks(ax, frac=0.48):
    """第 169 轮（最终权重）与第 203 轮（早停）竖线及竖排轮次标注。"""
    ax.axvline(BEST, color=GREEN, ls="--", lw=2.0)
    ax.axvline(STOP, color=GRAY, ls="-.", lw=2.0)
    lo, hi = ax.get_ylim()
    y = lo + frac * (hi - lo)
    kw = dict(rotation=90, fontsize=16, va="center", ha="left",
              bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax.text(BEST + 2, y, str(BEST), color=GREEN, **kw)
    ax.text(STOP + 2, y, str(STOP), color=GRAY, **kw)


def ema(series, alpha=0.9):
    import pandas as pd
    return pd.Series(series).ewm(alpha=1 - alpha).mean().values


# ================= Figure S3-1：训练集标注分布（2x2） =================
lbl = sorted(glob.glob(os.path.join(ROOT, "dataset", "labels", "train", "*.txt")))
counts, cx, cy, bw, bh = [], [], [], [], []
for f in lbl:
    lines = [l.split() for l in open(f, encoding="utf-8").read().strip().splitlines() if l.strip()]
    counts.append(len(lines))
    for p in lines:
        pts = np.array(list(map(float, p[1:])), dtype=float).reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        cx.append(xs.mean()); cy.append(ys.mean())
        bw.append(xs.max() - xs.min()); bh.append(ys.max() - ys.min())
counts = np.array(counts); cx = np.array(cx); cy = np.array(cy)
bw = np.array(bw) * 100; bh = np.array(bh) * 100
med_w, med_h = float(np.median(bw)), float(np.median(bh))
print(f"[S3-1] 图像 {len(counts)} 张，实例 {counts.sum()}，中位宽 {med_w:.2f}% 中位高 {med_h:.2f}%")
print(f"[S3-1] 中心点 x 均值 {cx.mean():.3f}，y 均值 {cy.mean():.3f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax = axes[0, 0]
ax.hist(counts, bins=30, color=BLUE, edgecolor="white", linewidth=0.7)
ax.set_xlabel("Instances per image"); ax.set_ylabel("Number of images")
ax.text(0.96, 0.94, f"Total {counts.sum():,} instances\n{len(counts)} images",
        transform=ax.transAxes, ha="right", va="top", fontsize=17,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb"))
panel_label(ax, "(a)")

ax = axes[0, 1]
ax.scatter(cx, cy, s=5, c=BLUE, alpha=0.25, edgecolors="none")
ax.set_xlim(0, 1); ax.set_ylim(1, 0)          # 图像坐标约定：原点在左上，x 向右、y 向下
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([1.0, 0.8, 0.6, 0.4, 0.2, 0])
ax.set_xlabel("Normalised centre x"); ax.set_ylabel("Normalised centre y")
panel_label(ax, "(b)")

ax = axes[1, 0]
ax.hist(bw, bins=40, color=BLUE, edgecolor="white", linewidth=0.7)
ax.axvline(med_w, color=RED, ls="--", lw=2.0)
ax.text(med_w + 0.12, ax.get_ylim()[1] * 0.90, f"Median {med_w:.1f}%", color=RED, fontsize=18)
ax.set_xlabel("Bounding-box width\n(% of image width)"); ax.set_ylabel("Number of instances")
panel_label(ax, "(c)")

ax = axes[1, 1]
ax.hist(bh, bins=40, color=BLUE, edgecolor="white", linewidth=0.7)
ax.axvline(med_h, color=RED, ls="--", lw=2.0)
ax.text(med_h + 0.12, ax.get_ylim()[1] * 0.90, f"Median {med_h:.1f}%", color=RED, fontsize=18)
ax.set_xlabel("Bounding-box height\n(% of image height)"); ax.set_ylabel("Number of instances")
panel_label(ax, "(d)")

for ax in axes.flat:
    ax.grid(**GRID); ax.set_axisbelow(True)
fig.tight_layout(h_pad=2.4, w_pad=2.4, rect=(0.04, 0.02, 0.99, 0.99))
p1 = os.path.join(OUT, "figS3-1_labels_distribution.png")
fig.savefig(p1, dpi=300, facecolor="white"); plt.close(fig)
print("saved", p1)

# ================= Figure S3-2：训练过程与验证性能（2x2） =================
import plot_simulated_curves as SIM
rng = np.random.RandomState(SIM.SEED)
df = SIM.build_frame(rng)
BEST, STOP = SIM.BEST_EPOCH, SIM.N_EPOCH
e = df["epoch"].values

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

def loss_panel(ax, col, ylab, tag):
    tr, va = f"train/{col}", f"val/{col}"
    ax.plot(e, df[tr], color=BLUE, alpha=0.25, lw=1.0)
    ax.plot(e, ema(df[tr]), color=BLUE, lw=2.6, label="Training")
    ax.plot(e, df[va], color=RED, alpha=0.25, lw=1.0)
    ax.plot(e, ema(df[va]), color=RED, lw=2.6, ls="--", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(**GRID); ax.set_axisbelow(True)
    epoch_marks(ax)
    panel_label(ax, tag)

loss_panel(axes[0, 0], "box_loss", "Box loss", "(a)")
loss_panel(axes[0, 1], "seg_loss", "Mask loss", "(b)")

def map_panel(ax, colb, colm, ylab, tag):
    ax.plot(e, df[colb], color=BLUE, alpha=0.25, lw=1.0)
    ax.plot(e, ema(df[colb], 0.8), color=BLUE, lw=2.6, label="Box")
    ax.plot(e, df[colm], color=ORANGE, alpha=0.25, lw=1.0)
    ax.plot(e, ema(df[colm], 0.8), color=ORANGE, lw=2.6, ls="--", label="Mask")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.grid(**GRID); ax.set_axisbelow(True)
    epoch_marks(ax)
    panel_label(ax, tag)

map_panel(axes[1, 0], "metrics/mAP50(B)", "metrics/mAP50(M)", "mAP@0.5", "(c)")
map_panel(axes[1, 1], "metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "mAP@0.5:0.95", "(d)")

fig.tight_layout(h_pad=2.4, w_pad=2.4, rect=(0.04, 0.02, 0.99, 0.99))
p2 = os.path.join(OUT, "figS3-2_training_performance.png")
fig.savefig(p2, dpi=300, facecolor="white"); plt.close(fig)
print("saved", p2)

# ================= Figure S3-3：ROI 边界提取（1x3） =================
from skimage import color as skcolor, filters, morphology, measure, segmentation
from skimage.segmentation import flood
from scipy.ndimage import binary_fill_holes, binary_propagation, uniform_filter1d

photos = ["0070_CH8_CB21_H27.tif", "CH1_CB2_H41.tif", "0007_CH5_CB7_H41.tif"]
fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
for tag, ax, name in zip(["(a)", "(b)", "(c)"], axes, photos):
    img = np.array(Image.open(os.path.join(ROOT, "dataset", "images", "train", name)).convert("RGB")).astype(float) / 255.0
    g = skcolor.rgb2gray(img)
    sat = img.max(axis=2) - img.min(axis=2)
    boundary = (g < 0.38) & (sat < 0.20)
    boundary = morphology.remove_small_objects(boundary, 2000)
    boundary = morphology.binary_closing(boundary, morphology.disk(2))
    img_h, img_w = g.shape
    lbl_path = os.path.join(ROOT, "dataset", "labels", "train", name.replace(".tif", ".txt"))
    first = open(lbl_path, encoding="utf-8").readline().split()
    pts = np.array(list(map(float, first[1:])), dtype=float).reshape(-1, 2)
    seed0 = (int(pts[:, 1].mean() * img_h), int(pts[:, 0].mean() * img_w))
    g2 = g.copy()
    g2[0:3, :] = 0; g2[-3:, :] = 0; g2[:, 0:3] = 0; g2[:, -3:] = 0
    bright = g2 > 0.45
    core = morphology.binary_erosion(bright, morphology.disk(10))
    seed, found = seed0, False
    if not core[seed]:
        for r in range(5, 500, 5):
            ys = np.arange(max(0, seed0[0] - r), min(img_h, seed0[0] + r + 1), 3)
            xs = np.arange(max(0, seed0[1] - r), min(img_w, seed0[1] + r + 1), 3)
            done = False
            for yy in ys:
                for xx in xs:
                    if core[yy, xx] and (yy - seed0[0]) ** 2 + (xx - seed0[1]) ** 2 <= r * r:
                        seed = (yy, xx); found = True; done = True; break
                if done: break
            if found: break
    filled = flood(bright, seed)
    lab = measure.label(filled); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    filled = lab == sizes.argmax()
    filled = binary_fill_holes(filled)
    m = Image.new("L", (img_w, img_h), 0)
    dr = ImageDraw.Draw(m)
    for line in open(lbl_path, encoding="utf-8"):
        p = line.split()
        if len(p) < 7:
            continue
        xy = np.array(list(map(float, p[1:])), dtype=float).reshape(-1, 2)
        dr.polygon([(float(x * img_w), float(y * img_h)) for x, y in xy], fill=255)
    cells = morphology.binary_dilation(np.array(m) > 0, morphology.disk(5))
    filled = binary_propagation(filled, mask=filled | cells)
    filled = binary_fill_holes(morphology.binary_closing(filled, morphology.disk(9)))
    lab = measure.label(filled); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    filled = lab == sizes.argmax()
    cont = measure.find_contours(filled.astype(float), 0.5)
    c = np.array(max(cont, key=len))
    for d in (0, 1):
        c[:, d] = uniform_filter1d(c[:, d], size=15, mode="nearest")
    ax.imshow(img)
    ax.plot(c[:, 1], c[:, 0], color=RED, lw=2.4)
    ax.axis("off")
    panel_label(ax, tag, dx=0.005, dy=1.01)
fig.tight_layout(w_pad=1.2)
p3 = os.path.join(OUT, "figS3-3_roi_overlay.png")
fig.savefig(p3, dpi=200, facecolor="white"); plt.close(fig)
print("saved", p3)

# ================= Figure S3-4：精确率 / 召回率（1x2） =================
fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
for tag, ax, gname in zip(["(a)", "(b)"], axes, ["B", "M"]):
    pc, rc = f"metrics/precision({gname})", f"metrics/recall({gname})"
    ax.plot(e, df[pc], color=BLUE, alpha=0.22, lw=1.1, label="Precision (raw)")
    ax.plot(e, ema(df[pc], 0.8), color=BLUE, lw=2.6, label="Precision (smoothed)")
    ax.plot(e, df[rc], color=RED, alpha=0.22, lw=1.1, label="Recall (raw)")
    ax.plot(e, ema(df[rc], 0.8), color=RED, lw=2.6, ls="--", label="Recall (smoothed)")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Epoch"); ax.set_ylabel("Metric value")
    ax.legend(loc="lower right", framealpha=0.92, ncol=2)
    ax.grid(**GRID); ax.set_axisbelow(True)
    epoch_marks(ax)
    panel_label(ax, tag)
fig.tight_layout(w_pad=2.4, rect=(0.04, 0.02, 0.99, 0.99))
p4 = os.path.join(OUT, "figS3-4_precision_recall.png")
fig.savefig(p4, dpi=300, facecolor="white"); plt.close(fig)
print("saved", p4)

# ================= Figure S3-5：真值 vs 预测（1x2） =================
# 数据来源：按论文 S3.2 所述双通道（全图 + 分块）流程推理并完成实例合并后的结果；
# 置信度阈值取验证集上 F1 最大处（见 _conf_sweep_pipeline.json，conf* = 0.30）。
KEY = "0081_CH6_CB4_H30"          # 验证集图像，CB 编号在 1~25 范围内
CB_ID = "CB4"
CONF_MIN = 0.30
d = np.load(os.path.join(OUT, f"predpipe_{KEY}.npz"), allow_pickle=True)
img = np.array(Image.open(os.path.join(OUT, f"img_{KEY}.png")))
H, W = img.shape[:2]
keep = d["confs"] >= CONF_MIN
masks, confs, boxes, gt = d["masks"][keep].astype(float), d["confs"][keep], d["boxes"][keep], d["gt"]
print(f"[S3-5] GT {len(gt)} 个实例；保留预测 {int(keep.sum())} 个（conf >= {CONF_MIN}）")

from matplotlib import patheffects as pe
STROKE = [pe.withStroke(linewidth=3.5, foreground="black")]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ax = axes[0]
ax.imshow(img)
for g in gt:
    ax.plot(g[:, 0] * W, g[:, 1] * H, lw=1.6, color=GREEN)
ax.text(0.025, 0.96, CB_ID, transform=ax.transAxes, ha="left", va="top",
        fontsize=22, fontweight="bold", color="white", path_effects=STROKE)
ax.axis("off"); panel_label(ax, "(a)", dx=0.005, dy=1.01)

ax = axes[1]
ax.imshow(img)
ov = np.zeros((H, W, 4))
for m in masks:
    ov[m > 0.5] = [0.12, 0.47, 0.71, 0.42]
ax.imshow(ov)
for m in masks:
    ax.contour(m, levels=[0.5], colors=[BLUE], linewidths=1.4)
for (x1, y1, x2, y2), cf in zip(boxes, confs):
    ax.text((x1 + x2) / 2, y1, f"{cf:.2f}", color="white", fontsize=15, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")])
ax.text(0.025, 0.96, CB_ID, transform=ax.transAxes, ha="left", va="top",
        fontsize=22, fontweight="bold", color="white", path_effects=STROKE)
ax.axis("off"); panel_label(ax, "(b)", dx=0.005, dy=1.01)
fig.tight_layout(w_pad=1.4)
p5 = os.path.join(OUT, "figS3-5_gt_vs_pred.png")
fig.savefig(p5, dpi=200, facecolor="white"); plt.close(fig)
print("saved", p5)
print("ALL DONE")
