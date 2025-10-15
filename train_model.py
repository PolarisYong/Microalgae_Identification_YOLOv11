from ultralytics import YOLO


def train_model():
    # 加载预训练模型
    model = YOLO('yolov8n-seg.pt')  # 使用 Nano 版本（轻量快速）

    model.train(
        data='data.yaml',
        device=0,           # GPU
        epochs=200,          # 训练轮次
        imgsz=1248,           # 输入图像尺寸
        batch=8,            # 批次大小
        pretrained=True,     # 使用预训练权重
        optimizer='AdamW',    # 优化器
        lr0=0.0005,           # 初始学习率
        augment=True,        # 数据增强
        mask_ratio=2,        # 掩码下采样率
        overlap_mask=True,    # 掩码是否重叠
        patience=20
    )


if __name__ == '__main__':
    train_model()