# -*- coding: utf-8 -*-
"""按论文所述双通道（全图+分块）+ 实例合并流程，在 44 张验证集图像上标定置信度阈值。
每个阈值下按置信度降序做一对一贪心匹配（掩膜 IoU >= 0.5 计 TP），输出 P/R/F1。
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
files = sorted(f for f in os.listdir(VAL_IMG) if f.lower().endswith(".tif"))

per_image = {}   # fn -> dict(gt=[masks], preds=[(conf, mask)])
n_tile_total = 0
for fi, fn in enumerate(files):
    img_bgr = cv2.imread(os.path.join(VAL_IMG, fn), cv2.IMREAD_COLOR)
    H, W = img_bgr.shape[:2]
    res, cfg, err, oom = M.predict_with_retry(model, img_bgr, mode="full")
    full = M.extract_instances_from_result(res[0], 0, 0) if res else []
    tiles, _ = M.predict_tiles(model, img_bgr)
    n_tile_total += len(tiles)
    merged = [i for i in M.merge_instances(full + tiles) if i.area_pixels >= M.MIN_MASK_AREA]

    def to_full(inst):
        canvas = np.zeros((H, W), bool)
        x1, y1, x2, y2 = inst.box
        y0, x0 = max(0, y1), max(0, x1)
        crop = inst.mask.astype(bool)
        ch, cw = crop.shape[:2]
        canvas[y0:y0 + ch, x0:x0 + cw] = crop[:H - y0, :W - x0]
        return canvas

    gt = []
    lp = os.path.join(VAL_LBL, fn.replace(".tif", ".txt"))
    for line in open(lp, encoding="utf-8"):
        s = line.split()
        if len(s) < 7:
            continue
        xy = np.array(list(map(float, s[1:])), dtype=float).reshape(-1, 2)
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).polygon([(float(x * W), float(y * H)) for x, y in xy], fill=255)
        gt.append(np.array(im) > 0)
    per_image[fn] = dict(gt=gt, preds=[(float(i.conf), to_full(i)) for i in merged])
    print(f"[{fi+1}/{len(files)}] {fn}: GT={len(gt)} 全图={len(full)} 分块={len(tiles)} 合并后={len(merged)}",
          flush=True)

n_gt = sum(len(v["gt"]) for v in per_image.values())
n_pred = sum(len(v["preds"]) for v in per_image.values())
print(f"\n验证集: {len(files)} 张图像, 真值 {n_gt} 个, 合并后预测 {n_pred} 个 (分块原始检出 {n_tile_total} 个)")

np.save(os.path.join(OUT, "_val_pipeline_preds.npy"),
        np.array([(fn, np.array([p[0] for p in v["preds"]], dtype=object)) for fn, v in per_image.items()],
                 dtype=object), allow_pickle=True)


def curve(conf_min):
    tp = fp = 0
    for fn, v in per_image.items():
        gts = v["gt"]
        cand = sorted(v["preds"], key=lambda x: -x[0])
        cand = [c for c in cand if c[0] >= conf_min]
        used = [False] * len(gts)
        for cf, m in cand:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                it = np.logical_and(m, g).sum()
                if it == 0:
                    continue
                u = m.sum() + g.sum() - it
                v_ = it / u
                if v_ > best:
                    best, bj = v_, j
            if best >= 0.5:
                used[bj] = True
                tp += 1
            else:
                fp += 1
    fn_ = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn_, p, r, f1


print("\n conf     TP    FP    FN   Precision  Recall     F1")
out = {}
for c in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
    tp, fp, fn_, p, r, f1 = curve(c)
    out[round(c, 2)] = dict(tp=tp, fp=fp, fn=fn_, P=round(p, 4), R=round(r, 4), F1=round(f1, 4))
    print(f" {c:.2f}  {tp:5d} {fp:5d} {fn_:5d}   {p:8.3f} {r:7.3f} {f1:7.3f}")
best = max(out, key=lambda c: out[c]["F1"])
print(f"\nF1 最优阈值 conf* = {best:.2f}  →  P={out[best]['P']:.3f}  R={out[best]['R']:.3f}  F1={out[best]['F1']:.3f}")
json.dump(dict(n_gt=n_gt, n_pred=n_pred, curve={str(k): v for k, v in out.items()}, best_conf=best),
          open(os.path.join(OUT, "_conf_sweep_pipeline.json"), "w"), indent=2)
print("DONE")
