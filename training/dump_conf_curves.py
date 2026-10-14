# -*- coding: utf-8 -*-
"""在验证集上重跑一次评估，导出 P/R/F1 随置信度阈值的变化（用于确定推理阈值）。"""
import json, os, sys
import numpy as np
from ultralytics import YOLO

ROOT = r"E:\pythonProject\Microalgae_Identification_YOLOv11"
W = os.path.join(ROOT, "runs", "segment", "microalgae_yolo26_lowmem", "weights", "best.pt")
m = YOLO(W)
r = m.val(data=os.path.join(ROOT, "data.yaml"), imgsz=896, split="val",
          plots=False, save_json=False, conf=0.001, iou=0.7, verbose=False)

box = r.box
out = {}
for name, arr in [("f1", box.f1_curve), ("p", box.p_curve), ("r", box.r_curve)]:
    out[name] = np.asarray(arr).ravel().tolist()
px = np.linspace(0, 1, len(out["f1"])).tolist()
out["px"] = px
out["mAP50"] = float(box.map50)
out["mAP"] = float(box.map)
out["mp"] = float(box.mp)
out["mr"] = float(box.mr)
json.dump(out, open(os.path.join(ROOT, "training_figures", "s3_revised", "_conf_curves.json"), "w"))

f1 = np.array(out["f1"]); p = np.array(out["p"]); rr = np.array(out["r"])
k = int(np.argmax(f1))
print("mAP50=%.4f mAP50-95=%.4f  P=%.4f R=%.4f" % (out["mAP50"], out["mAP"], out["mp"], out["mr"]))
print("F1 最大: conf=%.3f  F1=%.3f  P=%.3f  R=%.3f" % (px[k], f1[k], p[k], rr[k]))
print("\n conf    P      R      F1")
for c in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    i = int(round(c * (len(px) - 1)))
    print(" %.2f  %.3f  %.3f  %.3f" % (px[i], p[i], rr[i], f1[i]))
print("\nP>=0.95 的最小 conf:", next((round(px[i], 3) for i in range(len(px)) if p[i] >= 0.95 and i > 0), None))
