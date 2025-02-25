import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models import ResNet18_Weights

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

        # 每个epoch分为训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()   # 设置模型为评估模式
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 每个epoch结束后输出信息
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录最佳模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # 学习率调整
        scheduler.step()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# 训练模型
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
