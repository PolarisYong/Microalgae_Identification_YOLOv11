# -*- coding: utf-8 -*-
"""在 yolo26 环境下运行最终权重推理，导出 Figure S3-5 所需数据（图像 + 预测掩膜/框/置信度）。"""
import os, sys, json
import numpy as np
from PIL import Image

ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
OUT = os.path.join(ROOT, "training_figures", "s3_revised")
os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "weights", "best.pt")

CAND = {
    "0081_CH6_CB4_H30.tif": "CB4",
    "CH1_CB2_H16.tif": "CB2",
    "CH2_CB1_H35.tif": "CB1",
    "0047_CH5_CB10_H33.tif": "CB10",
    "CH1_CB2_H2.tif": "CB2",
}

from ultralytics import YOLO
model = YOLO(WEIGHTS)

meta = {}
for fn, cbid in CAND.items():
    p = os.path.join(ROOT, "dataset", "images", "val", fn)
    img = np.array(Image.open(p).convert("RGB"))
    H, W = img.shape[:2]
    r = model.predict(img, imgsz=896, conf=0.20, iou=0.45, max_det=300,
                      retina_masks=True, verbose=False)[0]
    masks = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0, H, W), np.uint8)
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    gt = []
    lp = os.path.join(ROOT, "dataset", "labels", "val", fn.replace(".tif", ".txt"))
    for line in open(lp, encoding="utf-8"):
        s = line.split()
        if len(s) >= 7:
            gt.append(np.array(list(map(float, s[1:])), dtype=float).reshape(-1, 2))
    key = fn.replace(".tif", "")
    np.savez_compressed(
        os.path.join(OUT, f"pred_{key}.npz"),
        masks=masks.astype(np.uint8), confs=confs, boxes=boxes,
        gt=np.array(gt, dtype=object) if gt else np.zeros((0,), dtype=object),
        shape=np.array([H, W]),
    )
    Image.fromarray(img).save(os.path.join(OUT, f"img_{key}.png"))
    meta[key] = dict(cbid=cbid, n_gt=len(gt), n_pred=int(len(confs)),
                     conf_min=float(confs.min()) if len(confs) else None,
                     conf_max=float(confs.max()) if len(confs) else None)
    print(key, meta[key])

json.dump(meta, open(os.path.join(OUT, "_inference_meta.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("DONE")
