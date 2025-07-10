import cv2
import numpy as np

# 图片路径和标注路径
image_path = "./datasets/yolo_data1/images/train/0737.png"
label_path = "./datasets/yolo_data1/labels/train/0737.txt"

# 类别名称（根据你的 my_dataset.yaml 文件中的 names 字段修改）
class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# 读取图片
image = cv2.imread(image_path)
if image is None:
    print(f"无法读取图片: {image_path}")
    exit()

# 获取图片的宽度和高度
h, w = image.shape[:2]

# 读取标注文件
with open(label_path, 'r') as f:
    lines = f.readlines()

# 遍历标注信息
for line in lines:
    # 解析标注信息
    class_id, x_center, y_center, width, height = map(float, line.strip().split())
    class_id = int(class_id)

    # 将相对坐标转换为绝对坐标
    x_center, y_center, width, height = int(x_center * w), int(y_center * h), int(width * w), int(height * h)

    # 计算边界框的左上角和右下角坐标
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # 绘制边界框和类别名称
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图片
cv2.imshow("Image with Annotations", image)
cv2.waitKey(0)  # 按任意键继续
cv2.destroyAllWindows()