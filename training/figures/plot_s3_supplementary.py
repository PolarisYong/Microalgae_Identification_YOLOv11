# -*- coding: utf-8 -*-
"""生成 S3 支撑材料补充图：Figure S6 标注分布 / Figure S7 ROI 叠加 / Figure S8 真值-预测对照。
风格与 plot_training_curves.py 保持一致（微软雅黑 + Times New Roman, dpi 200）。"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

# ---------- 全局样式（与 plot_training_curves.py 一致） ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["mathtext.fontset"] = "stix"

BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"
ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
OUT = os.path.join(ROOT, "training_figures", "s3_supplementary")
os.makedirs(OUT, exist_ok=True)

# ================= Figure S6：训练集标注分布 =================
lbl_files = sorted(glob.glob(os.path.join(ROOT, "dataset", "labels", "train", "*.txt")))
counts, cx, cy, bw, bh = [], [], [], [], []
for f in lbl_files:
    lines = [l.split() for l in open(f, encoding="utf-8").read().strip().splitlines() if l.strip()]
    counts.append(len(lines))
    for p in lines:
        pts = np.array(list(map(float, p[1:])), dtype=float).reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        cx.append(xs.mean()); cy.append(ys.mean())
        bw.append(xs.max() - xs.min()); bh.append(ys.max() - ys.min())
counts = np.array(counts); cx = np.array(cx); cy = np.array(cy)
bw = np.array(bw) * 100; bh = np.array(bh) * 100  # 归一化坐标 → % 视野

fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
ax = axes[0]
ax.hist(counts, bins=30, color=BLUE, edgecolor="white", linewidth=0.6)
ax.set_xlabel("每张图像实例数"); ax.set_ylabel("图像数量")
ax.set_title("(a) 实例数分布", fontsize=11)
ax.text(0.97, 0.95, f"总计 {counts.sum():,} 实例\n{len(counts)} 张图像",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

ax = axes[1]
ax.scatter(cx, cy, s=3, c=BLUE, alpha=0.25, edgecolors="none")
ax.set_xlim(0, 1); ax.set_ylim(1, 0)  # 图像坐标习惯：原点在左上，x 向右、y 向下，与实际视野方向一致
ax.set_xlabel("中心点 x（归一化）"); ax.set_ylabel("中心点 y（归一化）")
ax.set_title("(b) 中心点空间分布", fontsize=11)

ax = axes[2]
ax.hist(bw, bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
ax.axvline(np.median(bw), color=RED, ls="--", lw=1.2)
ax.text(np.median(bw) + 0.15, ax.get_ylim()[1] * 0.92, f"中位数 {np.median(bw):.1f}%",
        color=RED, fontsize=9)
ax.set_xlabel("边界框宽度（占视野宽度 %）"); ax.set_ylabel("实例数量")
ax.set_title("(c) 目标宽度分布", fontsize=11)

ax = axes[3]
ax.hist(bh, bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
ax.axvline(np.median(bh), color=RED, ls="--", lw=1.2)
ax.text(np.median(bh) + 0.15, ax.get_ylim()[1] * 0.92, f"中位数 {np.median(bh):.1f}%",
        color=RED, fontsize=9)
ax.set_xlabel("边界框高度（占视野高度 %）"); ax.set_ylabel("实例数量")
ax.set_title("(d) 目标高度分布", fontsize=11)

for ax in axes:
    ax.tick_params(labelsize=9)
fig.tight_layout()
p6 = os.path.join(OUT, "figS6_labels_distribution.png")
fig.savefig(p6, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("S6 saved:", p6)

# ================= Figure S7：三密度腔室照片 + ROI 叠加 =================
from skimage import color, filters, morphology, measure, segmentation
from scipy.ndimage import binary_fill_holes

photos = [
    ("0070_CH8_CB21_H27.tif", "(a) 低密度（2 个实例）"),
    ("CH1_CB2_H41.tif",       "(b) 中密度（82 个实例）"),
    ("0007_CH5_CB7_H41.tif",  "(c) 高密度（224 个实例）"),
]
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
for ax, (name, label) in zip(axes, photos):
    img = np.array(Image.open(os.path.join(ROOT, "dataset", "images", "train", name)).convert("RGB")).astype(float) / 255.0
    g = color.rgb2gray(img)
    sat = img.max(axis=2) - img.min(axis=2)  # 色度：细胞偏绿（高），边界线为灰黑（低）
    # 仅取低饱和度的暗色像素 = 腔室边界线；绿色细胞簇被排除，成为内部孔洞随后被填充
    boundary = (g < 0.38) & (sat < 0.20)
    boundary = morphology.remove_small_objects(boundary, 2000)
    boundary = morphology.binary_closing(boundary, morphology.disk(2))
    # 泛洪提取腔室内部：种子点取该图首个标注多边形质心（必在腔室内），
    # 暗色边界线 + 图像边框作为屏障，亮区域为可通行区
    from skimage.segmentation import flood
    img_h, img_w = g.shape
    first = open(os.path.join(ROOT, "dataset", "labels", "train", name.replace(".tif", ".txt")),
                 encoding="utf-8").readline().split()
    pts = np.array(list(map(float, first[1:])), dtype=float).reshape(-1, 2)
    seed0 = (int(pts[:, 1].mean() * img_h), int(pts[:, 0].mean() * img_w))
    g2 = g.copy()
    g2[0:3, :] = 0; g2[-3:, :] = 0; g2[:, 0:3] = 0; g2[:, -3:] = 0  # 边框屏障
    bright = g2 > 0.45
    # 腐蚀排除细胞内部小暗斑，种子须处于大块亮区内部
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
    lab = measure.label(filled)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    filled = lab == sizes.argmax()
    filled = binary_fill_holes(filled)  # 残余暗斑孔洞，回填
    # 修正：密集细胞簇贴壁时亮区泛洪会向内凹陷。以标注多边形（全部细胞）为
    # 可穿越区做条件传播，使区域穿过细胞簇直达腔室壁，再闭运算填补细缝
    from scipy.ndimage import binary_propagation
    m = Image.new("L", (img_w, img_h), 0)
    dr = ImageDraw.Draw(m)
    lbl_path = os.path.join(ROOT, "dataset", "labels", "train", name.replace(".tif", ".txt"))
    for line in open(lbl_path, encoding="utf-8"):
        p = line.split()
        if len(p) < 7:
            continue
        xy = np.array(list(map(float, p[1:])), dtype=float).reshape(-1, 2)
        dr.polygon([(float(x * img_w), float(y * img_h)) for x, y in xy], fill=255)
    cells = morphology.binary_dilation(np.array(m) > 0, morphology.disk(5))
    filled = binary_propagation(filled, mask=filled | cells)
    filled = binary_fill_holes(morphology.binary_closing(filled, morphology.disk(9)))
    lab = measure.label(filled)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    filled = lab == sizes.argmax()
    cont = measure.find_contours(filled.astype(float), 0.5)
    cont = [max(cont, key=len)]  # 主轮廓（腔室+连接颈）
    # 端点不回绕的滑动平均平滑
    from scipy.ndimage import uniform_filter1d
    c = np.array(cont[0])
    k = 15
    for d in (0, 1):
        c[:, d] = uniform_filter1d(c[:, d], size=k, mode="nearest")
    cont = [c]
    ax.imshow(img)
    for c in cont:
        ax.plot(c[:, 1], c[:, 0], color=RED, lw=1.8)
    ax.set_title(label, fontsize=11)
    ax.axis("off")
fig.tight_layout()
p7 = os.path.join(OUT, "figS7_roi_overlay.png")
fig.savefig(p7, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("S7 saved:", p7)

# ================= Figure S8：验证集真值 vs 预测对照 =================
lab_img = Image.open(os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "val_batch1_labels.jpg"))
pred_img = Image.open(os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "val_batch1_pred.jpg"))
W, H = lab_img.size
box = (0, H // 2, W // 2, H)  # 批内第二幅（0037_CH2_CB37_H26）
a = lab_img.crop(box); b = pred_img.crop(box)
gap = 12
canvas = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height)), "white")
canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + gap, 0))
draw = ImageDraw.Draw(canvas)
p8 = os.path.join(OUT, "figS8_gt_vs_pred.png")
canvas.save(p8, dpi=(200, 200))
print("S8 saved:", p8)
print("ALL DONE")
