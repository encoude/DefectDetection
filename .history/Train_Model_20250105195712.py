#模型训练

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 自定义数据集
class DefectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # 存储数据的目录路径
        self.transform = transform  # 数据预处理方法（可选）
        self.images = []  # 存储图像路径的列表
        self.labels = []  # 存储图像对应标签的列表
        # 遍历'缺陷'和'无缺陷'两个文件夹
        for label in ['defect', 'no_defect']:
            label_dir = os.path.join(data_dir, label)  # 获取当前类别的文件夹路径
            for img_name in os.listdir(label_dir):  # 遍历文件夹中的图像文件
                if img_name.endswith('.bmp'):  # 如果是bmp文件
                    self.images.append(os.path.join(label_dir, img_name))  # 添加图像路径
                    self.labels.append(0 if label == 'no_defect' else 1)  # 给每个图像打标签，0表示无缺陷，1表示有缺陷

    def __len__(self):
        return len(self.images)  # 返回数据集中的图像数量

    def __getitem__(self, idx):
        img_path = self.images[idx]  # 获取当前图像的路径
        label = self.labels[idx]  # 获取当前图像的标签
        img = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
        if self.transform:
            img = self.transform(img)  # 如果提供了预处理方法，则应用到图像
        return img, label  # 返回图像和对应的标签


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

# 加载预训练的 ResNet18 模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 修改最后的全连接层
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 类: defect 和 no_defect

# 将模型转移到 GPU (如果可用)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')
    
    # 保存模型
    torch.save(best_model_wts, 'best_model.pth')
    print("模型已保存至 best_model.pth")

    model.load_state_dict(best_model_wts)
    return model


# 训练模型
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
