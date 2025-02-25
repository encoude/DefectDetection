import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 设备类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像
img_path = r'path_to_image.jpg'  # 替换为你的图像路径
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# 获取ResNet18的特征图（假设我们提取的是最后一层卷积层的输出）
def get_feature_map(model, img_tensor):
    # 获取特定层的特征图
    features = []
    
    def hook_fn(module, input, output):
        features.append(output)
    
    # 注册hook函数到ResNet18的某一层
    layer = model.layer4[1].conv2  # 选择合适的层，这里是layer4的第二个卷积层
    hook = layer.register_forward_hook(hook_fn)
    
    # 执行一次前向传播，获取特征图
    model(img_tensor)
    
    # 移除hook
    hook.remove()
    
    return features[0]

# 提取特征图
feature_map = get_feature_map(model, img_tensor)

# 可视化特征图的第一通道
feature_map = feature_map.squeeze(0).cpu().detach().numpy()  # 移除batch维度并转移到CPU
plt.imshow(feature_map[0], cmap='viridis')
plt.colorbar()
plt.show()
