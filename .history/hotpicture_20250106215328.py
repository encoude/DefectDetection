import os
import config
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
device = torch.device(config.device)

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
                if img_name.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
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
    model.eval()  # 保持模型在评估模式
    img_tensor = img_tensor.requires_grad_(True)

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    final_conv_layer = model.layer4[1].conv2
    gradient = None
    activation_map = None
    
    def save_gradient(grad):
        nonlocal gradient
        gradient = grad
    
    def save_activation_map(module, input, output):
        nonlocal activation_map
        activation_map = output
    
    hook_activations = final_conv_layer.register_forward_hook(save_activation_map)
    hook_gradients = final_conv_layer.register_backward_hook(save_gradient)
    
    output = model(img_tensor)
    target_class = output.argmax(dim=1)
    
    # 确保梯度计算开启
    model.zero_grad()
    with torch.enable_grad():  # 强制启用梯度计算
        target = output[0, target_class]
        target.backward()

    gradients = gradient[0]
    activations = activation_map[0]

    if activations is None or gradients is None:
        print("Failed to capture activations or gradients.")
        return None

    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * activations, dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (224, 224))

    return cam
    img_tensor = img_tensor.requires_grad_(True)

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 转换为4D张量：[1, 3, 224, 224]

    final_conv_layer = model.layer4[1].conv2  # 确认这是最后的卷积层
    gradient = None
    activation_map = None
    
    def save_gradient(grad):
        nonlocal gradient  # 使用 nonlocal 而非 global
        gradient = grad
    
    def save_activation_map(module, input, output):
        nonlocal activation_map
        activation_map = output
    
    hook_activations = final_conv_layer.register_forward_hook(save_activation_map)
    hook_gradients = final_conv_layer.register_backward_hook(save_gradient)
    
    output = model(img_tensor)
    target_class = output.argmax(dim=1)

    model.zero_grad()
    output[:, target_class].backward()  # 仅对目标类别反向传播

    gradients = gradient[0]  # 获取梯度
    activations = activation_map[0]  # 获取激活图

    # 防止梯度或激活图为空
    if activations is None or gradients is None:
        print("Failed to capture activations or gradients.")
        return None

    # 计算加权特征图
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * activations, dim=0)
    cam = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  # 避免除以零

    cam = cam.cpu().detach().numpy()
    cam = np.uint8(255 * cam)

    # 插值到原始输入大小
    cam = cv2.resize(cam, (224, 224))

    return cam

# 将热力图与原始图像叠加
def overlay_heatmap(img, heatmap):
    # 将热力图应用到原始图像
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # 打印调试信息
    print(f"Overlay shape: {overlay.shape}")
    
    return overlay


# 显示热力图和原始图像
def show_heatmap(heatmap, img):
    plt.figure(figsize=(8, 8))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # 显示热力图
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet', alpha=0.6)  # 使用jet颜色映射并设置透明度
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


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

                    # 显示热力图
                    show_heatmap(heatmap, img)

                    # 将热力图叠加到原始图像
                    overlay = overlay_heatmap(img, heatmap)

                    # 保存带缺陷的图像和热力图
                    img_name = os.path.basename(img_path)
                    img.save(os.path.join(output_dir, f"{img_name}_original.bmp"))
                    heatmap_img = Image.fromarray(overlay)
                    heatmap_img.save(os.path.join(output_dir, f"{img_name}_heatmap.bmp"))

    print(f"Defect images and heatmaps have been saved to {output_dir}")

# 设置模型路径和输出文件夹
model_path = config.model_path
output_dir = config.output_dir

# 加载模型
model = load_model(model_path)

# 加载验证集
valid_dataset = DefectDataset(data_dir=config.valid_datasetPatn, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 检测并保存带缺陷的图像及其热力图
detect_and_save(model, valid_loader, output_dir)