from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(r"E:\pythonProject\Microalgae_Identification_YOLOv11")
DATA_YAML = PROJECT_ROOT / "data.yaml"
if not DATA_YAML.exists():
    fallback_yaml = PROJECT_ROOT / "dataset" / "data.yaml"
    if fallback_yaml.exists():
        DATA_YAML = fallback_yaml
RUNS_DIR = PROJECT_ROOT / "runs" / "segment"

# True = lower VRAM/RAM usage, False = higher accuracy / higher memory.
LOW_MEMORY_MODE = True

if LOW_MEMORY_MODE:
    BASE_MODEL = "yolo26n-seg.pt"
    TRAIN_CFG = {
        "data": str(DATA_YAML),
        "device": 0,
        "epochs": 300,
        "imgsz": 896,
        "batch": 1,
        "patience": 50,
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "cos_lr": True,
        "close_mosaic": 10,
        "overlap_mask": True,
        "amp": True,
        "cache": False,
        "workers": 2,
        "plots": True,
        "save": True,
        "project": str(RUNS_DIR),
        "name": "microalgae_yolo26_lowmem",
        "exist_ok": False,
        "seed": 42,
    }
else:
    BASE_MODEL = "yolo26s-seg.pt"
    TRAIN_CFG = {
        "data": str(DATA_YAML),
        "device": 0,
        "epochs": 300,
        "imgsz": 1024,
        "batch": 2,
        "patience": 50,
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "cos_lr": True,
        "close_mosaic": 10,
        "overlap_mask": True,
        "amp": True,
        "cache": False,
        "workers": 4,
        "plots": True,
        "save": True,
        "project": str(RUNS_DIR),
        "name": "microalgae_yolo26_seg",
        "exist_ok": False,
        "seed": 42,
    }


def train_model():
    model = YOLO(BASE_MODEL)
    model.train(**TRAIN_CFG)


if __name__ == "__main__":
    train_model()
