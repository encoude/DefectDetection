import os
import cv2
import numpy as np
import random

# 原始图像文件夹路径
source_folder = r"F:\python\AI\DefectDetection\2025.1.5\Data\marbled\all"  # 替换为你的文件夹路径
# 带缺陷图像保存路径
output_folder = r"F:\python\AI\DefectDetection\2025.1.5\Data\marbled\all"  # 替换为保存路径

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中的所有图片路径
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', '.bmp'))]

# 定义缺陷添加函数
def add_defect(image):
    height, width, channels = image.shape
    defect_type = random.choice(['noise', 'scratch', 'occlusion'])  # 随机选择缺陷类型
    
    if defect_type == 'noise':
        # 添加随机噪声
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        defective_image = cv2.add(image, noise)

    elif defect_type == 'scratch':
        # 模拟划痕：随机画一条线
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(1, 5)
        defective_image = image.copy()
        cv2.line(defective_image, (x1, y1), (x2, y2), color, thickness)

    elif defect_type == 'occlusion':
        # 模拟遮挡：添加随机矩形块
        x1, y1 = random.randint(0, width - 20), random.randint(0, height - 20)
        x2, y2 = random.randint(x1 + 10, min(x1 + 100, width)), random.randint(y1 + 10, min(y1 + 100, height))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        defective_image = image.copy()
        cv2.rectangle(defective_image, (x1, y1), (x2, y2), color, -1)

    return defective_image

# 遍历图像并生成缺陷图像
for file in image_files:
    # 读取图像
    image_path = os.path.join(source_folder, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {file}")
        continue

    # 添加缺陷
    defective_image = add_defect(image)

    # 保存图像
    base_name, ext = os.path.splitext(file)
    save_path = os.path.join(output_folder, f"{base_name}_defect{ext}")
    cv2.imwrite(save_path, defective_image)

    print(f"保存带缺陷的图像: {save_path}")
