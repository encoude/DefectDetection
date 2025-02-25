#构建模型
#ResNet18 是 残差网络（Residual Network） 中的一种变体，属于深度卷积神经网络（CNN）的架构
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# 加载预训练的 ResNet18 模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 修改最后的全连接层
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 类: defect 和 no_defect

# 将模型转移到 GPU (如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
