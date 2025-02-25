# #模型预测

# import os
# import torch
# from torchvision import transforms, models
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# from torchvision.models import ResNet18_Weights
# from PIL import Image

# #设备类型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 自定义数据集类（与训练时相同）
# class DefectDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.images = []
#         self.labels = []
#         for label in ['defect', 'no_defect']:
#             label_dir = os.path.join(data_dir, label)
#             for img_name in os.listdir(label_dir):
#                 if img_name.endswith('.bmp'):
#                     self.images.append(os.path.join(label_dir, img_name))
#                     self.labels.append(0 if label == 'no_defect' else 1)  # 0: no defect, 1: defect
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         label = self.labels[idx]
#         img = Image.open(img_path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# # 数据预处理（与训练时相同）
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 加载模型
# # 加载模型
# def load_model(model_path):
#     model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#     model.fc = nn.Linear(model.fc.in_features, 2)
#     model.load_state_dict(torch.load(model_path, weights_only=True))  # 修改为 weights_only=True
#     return model.to(device)


# # 使用模型对图像进行预测并保存带有缺陷的图像
# def detect_and_save(model, dataloader, output_dir):
#     model.eval()  # 设置模型为评估模式
#     os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             # 获取所有被预测为 defect 的图像
#             for idx, pred in enumerate(preds):
#                 if pred == 1:  # 1 代表 defect
#                     img_path = dataloader.dataset.images[idx]
#                     img = Image.open(img_path)
#                     # 保存带缺陷的图像
#                     img_name = os.path.basename(img_path)
#                     img.save(os.path.join(output_dir, img_name))

#     print(f"Defect images have been saved to {output_dir}")

# # 设置模型路径和输出文件夹
# model_path = r'F:\python\AI\DefectDetection\2025.1.5\best_model.pth'
# output_dir = r'F:\python\AI\DefectDetection\2025.1.5\defect_images'

# # 加载模型
# model = load_model(model_path)

# # 加载验证集
# valid_dataset = DefectDataset(data_dir=r"F:\项目软件\方昇\test3\val", transform=transform)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # 检测并保存带缺陷的图像
# detect_and_save(model, valid_loader, output_dir)


import os
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类（与训练时相同）
class DefectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label in ['defect', 'no_defect']:
            label_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.bmp'):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(0 if label == 'no_defect' else 1)  # 0: no defect, 1: defect
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 数据预处理（与训练时相同）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 修改为 weights_only=True
    return model.to(device)

# Grad-CAM 实现
def grad_cam(model, img_tensor):
    # 获取最后一个卷积层
    final_conv_layer = model.layer4[1].conv2
    gradient = None
    activation_map = None
    
    # 定义 hook 函数
    def save_gradient(grad):
        global gradient
        gradient = grad
    
    def save_activation_map(module, input, output):
        global activation_map
        activation_map = output
    
    # 注册 hook
    hook_activations = final_conv_layer.register_forward_hook(save_activation_map)
    hook_gradients = final_conv_layer.register_backward_hook(save_gradient)
    
    # 前向传播
    output = model(img_tensor)
    
    # 获取目标类的得分
    target_class = output.argmax(dim=1)
    
    # 反向传播，计算梯度
    model.zero_grad()
    target_class.backward()
    
    # 获取梯度和激活图
    gradients = gradient[0]
    activations = activation_map[0]
    
    # 对梯度进行全局平均池化
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    
    # 计算加权的特征图
    cam = torch.sum(weights * activations, dim=0)
    cam = F.relu(cam)
    
    # 对热力图进行归一化
    cam = cam - cam.min()
    cam = cam / cam.max()
    
    # 将热力图转换为0-255的图像
    cam = cam.cpu().detach().numpy()
    cam = np.uint8(255 * cam)
    
    return cam

# 将热力图与原始图像叠加
def overlay_heatmap(img, heatmap):
    # 将热力图应用到原始图像
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# 使用模型对图像进行预测并保存带有缺陷的图像和热力图
def detect_and_save(model, dataloader, output_dir):
    model.eval()  # 设置模型为评估模式
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 获取所有被预测为 defect 的图像
            for idx, pred in enumerate(preds):
                if pred == 1:  # 1 代表 defect
                    img_path = dataloader.dataset.images[idx]
                    img = Image.open(img_path)
                    img_tensor = inputs[idx].unsqueeze(0)

                    # 计算 Grad-CAM 热力图
                    heatmap = grad_cam(model, img_tensor)

                    # 检查 heatmap 是否有效
                    if heatmap is not None and heatmap.shape != (224, 224):
                        print(f"Invalid heatmap for {img_path}")
                        continue

                    # 将热力图叠加到原始图像
                    overlay = overlay_heatmap(img, heatmap)

                    # 保存带缺陷的图像和热力图
                    img_name = os.path.basename(img_path)
                    img.save(os.path.join(output_dir, f"{img_name}_original.bmp"))
                    heatmap_img = Image.fromarray(overlay)
                    heatmap_img.save(os.path.join(output_dir, f"{img_name}_heatmap.bmp"))

    print(f"Defect images and heatmaps have been saved to {output_dir}")


# 设置模型路径和输出文件夹
model_path = r'F:\python\AI\DefectDetection\2025.1.5\best_model.pth'
output_dir = r'F:\python\AI\DefectDetection\2025.1.5\defect_images'

# 加载模型
model = load_model(model_path)

# 加载验证集
valid_dataset = DefectDataset(data_dir=r"F:\python\AI\DefectDetection\2025.1.5\Data\val", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 检测并保存带缺陷的图像及其热力图
detect_and_save(model, valid_loader, output_dir)

