# -*- coding: utf-8 -*-
"""一次性完成两件事：
1) 在 44 张验证集图像上扫描置信度阈值，按掩膜 IoU>=0.5 匹配统计 TP/FP/FN，给出 P/R/F1 曲线与 F1 最优阈值；
2) 按论文所述双通道（全图+分块）流程推理 Figure S3-5 所用图像，并正确拼装整图掩膜。
"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
import cv2

ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
sys.path.insert(0, os.path.join(ROOT, "identification_of_microalgae"))
OUT = os.path.join(ROOT, "training_figures", "s3_revised")
WEIGHTS = os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "weights", "best.pt")
VAL_IMG = os.path.join(ROOT, "dataset", "images", "val")
VAL_LBL = os.path.join(ROOT, "dataset", "labels", "val")

import microalgae_dense_hybrid_00 as M
from ultralytics import YOLO

model = YOLO(WEIGHTS)


def gt_masks(fn, H, W):
    ms = []
    p = os.path.join(VAL_LBL, fn.replace(".tif", ".txt"))
    if not os.path.exists(p):
        return ms
    for line in open(p, encoding="utf-8"):
        s = line.split()
        if len(s) < 7:
            continue
        xy = np.array(list(map(float, s[1:])), dtype=float).reshape(-1, 2)
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).polygon([(float(x * W), float(y * H)) for x, y in xy], fill=255)
        ms.append(np.array(im) > 0)
    return ms


def iou(a, b):
    it = np.logical_and(a, b).sum()
    return it / (a.sum() + b.sum() - it) if it else 0.0


# ---------------- 1) 阈值扫描（单通道、imgsz=896，与训练验证一致） ----------------
files = sorted(f for f in os.listdir(VAL_IMG) if f.lower().endswith(".tif"))
records = []   # (conf, iou_best_matched) 逐图像记录
n_gt_total = 0
for fi, fn in enumerate(files):
    img = np.array(Image.open(os.path.join(VAL_IMG, fn)).convert("RGB"))
    H, W = img.shape[:2]
    gts = gt_masks(fn, H, W)
    n_gt_total += len(gts)
    r = model.predict(img, imgsz=896, conf=0.01, iou=0.45, max_det=300,
                      retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        continue
    pm = r.masks.data.cpu().numpy() > 0.5
    pm = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                   for m in pm])
    cf = r.boxes.conf.cpu().numpy()
    bx = r.boxes.xyxy.cpu().numpy()
    for i in range(len(cf)):
        best = max((iou(pm[i], g) for g in gts), default=0.0)
        records.append((float(cf[i]), float(best), fn, bx[i].tolist()))
    print(f"[{fi+1}/{len(files)}] {fn}: GT={len(gts)} 预测={len(cf)}", flush=True)

records.sort(key=lambda x: -x[0])
confs = np.array([r[0] for r in records])
ious = np.array([r[1] for r in records])

print(f"\n验证集图像 {len(files)} 张，真值实例 {n_gt_total} 个，候选预测 {len(records)} 个")
print("\n conf     TP    FP    FN   Precision  Recall     F1")
grid = [round(0.05 * k, 2) for k in range(1, 19)]
res = {}
for c in grid:
    k = confs >= c
    # 每个真值只能被匹配一次：按置信度从高到低贪心分配
    used = {}
    tp = 0
    for idx in np.where(k)[0]:
        conf, bi, fn, _ = records[idx]
        if bi < 0.5:
            continue
        # 该预测归属图像 fn；同一图像内同一真值不可重复计数
        key = (fn, round(bi, 6))
        if key in used:
            continue
        used[key] = True
        tp += 1
    tp = int(sum(1 for idx in np.where(k)[0] if ious[idx] >= 0.5))
    fp = int(k.sum()) - tp
    fn = n_gt_total - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / n_gt_total
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    res[c] = dict(tp=tp, fp=fp, fn=fn, P=round(p, 4), R=round(rc, 4), F1=round(f1, 4))
    print(f" {c:.2f}  {tp:5d} {fp:5d} {fn:5d}   {p:8.3f} {rc:7.3f} {f1:7.3f}")

best_c = max(res, key=lambda c: res[c]["F1"])
print(f"\nF1 最优阈值 conf* = {best_c:.2f}  (P={res[best_c]['P']:.3f}, R={res[best_c]['R']:.3f}, F1={res[best_c]['F1']:.3f})")
json.dump(dict(n_gt=n_gt_total, curve={str(k): v for k, v in res.items()}, best_conf=best_c),
          open(os.path.join(OUT, "_conf_sweep.json"), "w"), indent=2)

# ---------------- 2) Figure S3-5：双通道完整流程 ----------------
KEY = "0081_CH6_CB4_H30.tif"
img_bgr = cv2.imread(os.path.join(VAL_IMG, KEY), cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
res2, cfg, err, oom = M.predict_with_retry(model, img_bgr, mode="full")
full = M.extract_instances_from_result(res2[0], 0, 0) if res2 else []
tiles, _oom = M.predict_tiles(model, img_bgr)
merged = M.merge_instances(full + tiles)
merged = [i for i in merged if i.area_pixels >= M.MIN_MASK_AREA]


def to_full(inst):
    canvas = np.zeros((H, W), np.uint8)
    x1, y1, x2, y2 = inst.box
    y0, x0 = max(0, y1), max(0, x1)
    crop = inst.mask
    ch, cw = crop.shape[:2]
    canvas[y0:y0 + ch, x0:x0 + cw] = crop[:H - y0, :W - x0]
    return canvas


masks = np.stack([to_full(i) for i in merged]) if merged else np.zeros((0, H, W), np.uint8)
confs2 = np.array([i.conf for i in merged], dtype=np.float32)
boxes2 = np.array([i.box for i in merged], dtype=np.float32).reshape(-1, 4)
gt = []
for line in open(os.path.join(VAL_LBL, KEY.replace(".tif", ".txt")), encoding="utf-8"):
    s = line.split()
    if len(s) >= 7:
        gt.append(np.array(list(map(float, s[1:])), dtype=float).reshape(-1, 2))
k = KEY.replace(".tif", "")
np.savez_compressed(os.path.join(OUT, f"predpipe_{k}.npz"), masks=masks, confs=confs2,
                    boxes=boxes2, gt=np.array(gt, dtype=object), shape=np.array([H, W]))
Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(os.path.join(OUT, f"img_{k}.png"))

gts = gt_masks(KEY, H, W)
pm2 = masks > 0
print(f"\n[S3-5 双通道] GT={len(gts)}  合并后预测={len(confs2)}")
for i in range(len(confs2)):
    bi = max((iou(pm2[i], g) for g in gts), default=0.0)
    ys, xs = np.nonzero(pm2[i])
    print(f"  conf={confs2[i]:.3f}  面积={int(pm2[i].sum()):6d}  IoU_max={bi:.3f}  "
          f"中心=({xs.mean():.0f},{ys.mean():.0f})  bbox={boxes2[i].round(0).tolist()}")
print("DONE")
