#模型评估

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 自定义数据集
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

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = DefectDataset(data_dir=r"F:\项目软件\方昇\test3\train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = DefectDataset(data_dir=r"F:\项目软件\方昇\test3\val", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 检查是否有可用的GPU，若没有，则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
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
