import os
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt  # 导入绘图库

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
                if img_name.endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # 如果是bmp文件
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
    transforms.Resize((224, 224)),  # 将图像大小调整为224x224像素
    transforms.ToTensor(),  # 将图像从PIL格式转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差进行标准化
])

# 创建数据集和数据加载器
train_dataset = DefectDataset(data_dir=r"F:\python\AI\DefectDetection\2025.1.5\Data\train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = DefectDataset(data_dir=r"F:\python\AI\DefectDetection\2025.1.5\Data\val", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 加载预训练的 ResNet18 模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# 修改最后的全连接层
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 类: defect 和 no_defect

# 将模型转移到 GPU (如果可用)
device = torch.device('cuda')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于分类问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率设置为 0.001

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 每 7 个 epoch 调整一次学习率，gamma=0.1 表示学习率每次减少 10%

# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()  # 存储最佳模型权重
    best_acc = 0.0  # 初始最佳准确率

    # 记录每个 epoch 的损失和准确率
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:  # 训练和验证阶段
            if phase == 'train':
                model.train()  # 切换到训练模式
                dataloader = train_loader  # 训练集数据加载器
            else:
                model.eval()  # 切换到验证模式
                dataloader = valid_loader  # 验证集数据加载器

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:  # 遍历每一个批次
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到设备（GPU 或 CPU）
                optimizer.zero_grad()  # 清除梯度

                with torch.set_grad_enabled(phase == 'train'):  # 训练阶段启用梯度计算，验证阶段不计算梯度
                    outputs = model(inputs)  # 计算模型的输出
                    _, preds = torch.max(outputs, 1)  # 获取预测标签
                    loss = criterion(outputs, labels)  # 计算损失

                    if phase == 'train':  # 训练阶段进行反向传播
                        loss.backward()  # 计算梯度
                        optimizer.step()  # 更新模型参数

                running_loss += loss.item() * inputs.size(0)  # 累加损失
                running_corrects += torch.sum(preds == labels.data)  # 累加正确的预测数量

            epoch_loss = running_loss / len(dataloader.dataset)  # 计算每个epoch的平均损失
            epoch_acc = running_corrects.double() / len(dataloader.dataset)  # 计算每个epoch的准确率

            # 记录每个epoch的损失和准确率
            if phase == 'train':
                train_losses.append(epoch_loss)  # 保存损失
                train_accuracies.append(epoch_acc.item())  # 保存准确率，并确保是 Python float 类型
            else:
                valid_losses.append(epoch_loss)  # 保存损失
                valid_accuracies.append(epoch_acc.item())  # 保存准确率，并确保是 Python float 类型

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型权重
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step()  # 调整学习率

    # 绘制损失和准确率图表
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # 绘制训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), valid_losses, label='Valid Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), valid_accuracies, label='Valid Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(config.folder_name):
        os.makedirs(config.folder_name)

    # 保存图表到文件
    plt.savefig(os.path.join(config.folder_name, 'training_results.png'))
    # plt.savefig('training_results.png')  # 图表保存为 'training_results.png'
    print("训练过程的图表已保存为 training_results.png")

    return model

# 保存模型权重
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

# 训练模型
model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)

# 保存训练后的模型
model_save_path = os.path.join(config.folder_name, 'trained_model.pth')
save_model(model, model_save_path)