# training —— 模型训练、标注与论文配图

非主流程，仅在需要重新训练模型或重绘论文配图时使用。

| 路径 | 用途 |
|---|---|
| `train_model.py` | YOLO26-seg 分割模型训练（输出到 `runs/segment/`） |
| `dump_conf_curves.py` | `model.val()` 导出 F1/P/R 随置信度变化曲线 |
| `dataset_label/` | LabelMe JSON ↔ YOLO TXT 互转、按清单挑图、train/val 划分 |
| `figures/` | 训练曲线、S3 补充图、kLa 表征图等论文配图脚本 |

## 与识别主流程的关系

`microalgae_identification/detection/detect_batch.py` 默认加载
`runs/segment/train4/weights/best.pt`（已在版本库中）。重新训练后，
在批次配置里用 `model.weights` 指向新权重即可，无需改脚本：

```yaml
model:
  weights: "runs/segment/<新训练目录>/weights/best.pt"
```

## 环境

依赖与识别主流程相同，见仓库根 `requirements.txt`（含 `torch`、`ultralytics`）。
训练需要 CUDA 环境。
