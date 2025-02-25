import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)

# 修改最后的全连接层
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 类: defect 和 no_defect

# 将模型转移到 GPU (如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
