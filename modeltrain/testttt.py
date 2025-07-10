from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.yaml")  # 从 YAML 构建新模型
model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 从 YAML 构建并传递权重

# 训练模型
results = model.train(data="yolofish.yaml", epochs=100, imgsz=640)


