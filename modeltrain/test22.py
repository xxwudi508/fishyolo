from ultralytics import YOLO

# Load a model
model = YOLO("/home/data/home/usr05/ELF/AI3588-master/yolo11/ultralytics_yolo11/runs/detect/train14/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["datasets/yolo_data1/images/train/0011.png", "datasets/yolo_data1/images/train/0611.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk