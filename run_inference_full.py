# -*- coding: utf-8 -*-
"""按论文所述双通道（全图+分块）推理流程 + 实例合并，导出 Figure S3-5 所需结果。"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image

ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
sys.path.insert(0, os.path.join(ROOT, "identification_of_microalgae"))
OUT = os.path.join(ROOT, "training_figures", "s3_revised")
os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "weights", "best.pt")

import microalgae_dense_hybrid_00 as M
from ultralytics import YOLO

model = YOLO(WEIGHTS)

CAND = {"CH1_CB2_H2.tif": "CB2", "CH1_CB2_H16.tif": "CB2", "0081_CH6_CB4_H30.tif": "CB4"}
meta = {}
for fn, cbid in CAND.items():
    path = os.path.join(ROOT, "dataset", "images", "val", fn)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    H, W = img_bgr.shape[:2]

    # 全图通道
    res, cfg, err, oom = M.predict_with_retry(model, img_bgr, mode="full")
    full = M.extract_instances_from_result(res[0], 0, 0) if res else []
    # 分块通道
    tiles, _oom = M.predict_tiles(model, img_bgr)
    allinst = full + tiles
    merged = M.merge_instances(allinst)
    merged = [i for i in merged if i.area_pixels >= M.MIN_MASK_AREA]

    def to_full(inst):
        """Instance.mask 是仅覆盖其包围盒的局部掩膜，需按 box 位置贴回整图坐标系。"""
        canvas = np.zeros((H, W), np.uint8)
        x1, y1, x2, y2 = inst.box
        y0, x0 = max(0, y1), max(0, x1)
        crop = inst.mask
        ch, cw = crop.shape[:2]
        canvas[y0:y0 + ch, x0:x0 + cw] = crop[:H - y0, :W - x0]
        return canvas

    masks = np.stack([to_full(i) for i in merged]) if merged else np.zeros((0, H, W), np.uint8)
    confs = np.array([i.conf for i in merged], dtype=np.float32)
    boxes = np.array([i.box for i in merged], dtype=np.float32).reshape(-1, 4)
    gt = []
    for line in open(os.path.join(ROOT, "dataset", "labels", "val", fn.replace(".tif", ".txt")), encoding="utf-8"):
        s = line.split()
        if len(s) >= 7:
            gt.append(np.array(list(map(float, s[1:])), dtype=float).reshape(-1, 2))

    key = fn.replace(".tif", "")
    np.savez_compressed(os.path.join(OUT, f"predfull_{key}.npz"), masks=masks, confs=confs,
                        boxes=boxes, gt=np.array(gt, dtype=object), shape=np.array([H, W]))
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(os.path.join(OUT, f"img_{key}.png"))
    meta[key] = dict(cbid=cbid, n_gt=len(gt), n_full=len(full), n_tile=len(tiles),
                     n_merged=len(merged), conf=confs.round(3).tolist())
    print(key, meta[key])

json.dump(meta, open(os.path.join(OUT, "_inference_full_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("DONE")
