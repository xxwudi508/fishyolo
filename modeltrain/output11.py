from ultralytics import YOLO
model = YOLO("/home/data/home/usr05/ELF/AI3588-master/yolo11/ultralytics_yolo11/runs/detect/train14/weights/best.pt")
model.export(format='rknn') # 实际导出为onnx格式