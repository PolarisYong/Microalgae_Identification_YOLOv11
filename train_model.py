from ultralytics import YOLO


def train_model():
    # 加载预训练模型
    model = YOLO('yolov8s-seg.pt')  # 使用 Nano 版本（轻量快速）

    model.train(
        data='data.yaml',
        device=0,           # GPU
        epochs=200,          # 训练轮次
        imgsz=640,           # 输入图像尺寸
        batch=4,  # 不用改
        half=True,
        patience=30,  # 从20增加到30，给模型更多收敛时间
        lr0=0.0003,  # 学习率稍微降低一点，防止过拟合
        weight_decay=0.001,  # 增加权重衰减，正则化
        dropout=0.1,  # 加一点dropout，防止过拟合
        overlap_mask=True,
        augment=True,
        plots=True,
        save=True
    )


if __name__ == '__main__':
    train_model()