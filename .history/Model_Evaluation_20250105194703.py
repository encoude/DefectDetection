import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

# 检查是否有可用的GPU，若没有，则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)  # 将模型转移到设备上

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据转移到设备上
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Evaluation Accuracy: {accuracy:.4f}')

# 模型路径
model_path = r'F:\python\AI\DefectDetection\2025.1.5\best_model.pth'
model = load_model(model_path)

# 验证集加载
valid_dataset = DefectDataset(data_dir=r"F:\项目软件\方昇\test3\val", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 评估模型在验证集上的表现
evaluate_model(model, valid_loader)
