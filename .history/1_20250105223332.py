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

# 加载训练后的模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 类: defect 和 no_defect
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model = model.to('cuda')
    model.eval()  # 设置模型为评估模式
    return model

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

    # 打印检查热力图
    print(f"CAM shape: {cam.shape}, min: {cam.min().item()}, max: {cam.max().item()}")
    
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

# 显示和保存热力图
def show_and_save_heatmap(heatmap, img, img_path, output_dir):
    # 显示热力图
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

    # 保存图像和热力图
    img_name = os.path.basename(img_path)
    img.save(os.path.join(output_dir, f"{img_name}_original.bmp"))
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img.save(os.path.join(output_dir, f"{img_name}_heatmap.bmp"))
    print(f"Saved {img_name}_original.bmp and {img_name}_heatmap.bmp")

# 预测并保存带缺陷的图像和热力图
def predict_and_save(model, dataloader, output_dir):
    model.eval()  # 设置模型为评估模式
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to('cuda')
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

                    # 将热力图叠加到原始图像
                    overlay = overlay_heatmap(img, heatmap)

                    # 显示并保存图像和热力图
                    show_and_save_heatmap(overlay, img, img_path, output_dir)

# 预测并保存缺陷图和热力图
model_path = config.model_path
output_dir = config.output_dir

# 加载模型
model = load_model(model_path)

# 加载验证集
valid_dataset = DefectDataset(data_dir=config.valid_datasetPatn, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 进行预测并保存图像和热力图
predict_and_save(model, valid_loader, output_dir)
