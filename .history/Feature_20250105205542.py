import torch
import torch.nn as nn  # 确保导入了 nn 模块
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # 修改全连接层以适应二分类
model.to(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像
img_path = r'F:\python\AI\DefectDetection\2025.1.5\defect_images\Pic_2025_01_03_105101_12.bmp'  # 替换为你的图像路径
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# 确保输入张量的 requires_grad 属性为 True
img_tensor.requires_grad_()  # 需要计算梯度

# Grad-CAM
def grad_cam(model, img_tensor):
    # 注册 hook 获取最后卷积层的特征图
    final_conv_layer = model.layer4[1].conv2  # 选择卷积层
    gradient = None
    activation_map = None
    
    # 定义 hook 函数
    def save_gradient(grad):
        global gradient
        gradient = grad
    
    def save_activation_map(module, input, output):
        global activation_map
        activation_map = output
    
    # 注册钩子
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
    
    # 可视化
    cam = cam.cpu().detach().numpy()
    cam = np.uint8(255 * cam)
    
    # 将热力图叠加到原始图像上
    img = np.array(img)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return overlay

# 使用Grad-CAM提取图像中的缺陷区域
output_image = grad_cam(model, img_tensor)

# 显示结果
plt.imshow(output_image)
plt.show()
