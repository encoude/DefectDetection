#模型预测
import os
import torch
import config
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models import ResNet18_Weights
from PIL import Image

#设备类型
device = torch.device(config.device)

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

# 加载模型
# 加载模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 修改为 weights_only=True
    return model.to(device)


# 使用模型对图像进行预测并保存带有缺陷的图像
def detect_and_save(model, dataloader, output_dir):
    model.eval()  # 设置模型为评估模式
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 获取所有被预测为 defect 的图像
            for idx, pred in enumerate(preds):
                if pred == 1:  # 1 代表 defect
                    img_path = dataloader.dataset.images[idx]
                    img = Image.open(img_path)
                    # 保存带缺陷的图像
                    img_name = os.path.basename(img_path)
                    img.save(os.path.join(output_dir, img_name))

    print(f"Defect images have been saved to {output_dir}")

# 设置模型路径和输出文件夹
model_path = config.model_path
output_dir = config.output_dir

# 加载模型
model = load_model(model_path)

# 加载验证集
valid_dataset = DefectDataset(data_dir=config.valid_datasetPatn, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 检测并保存带缺陷的图像
detect_and_save(model, valid_loader, output_dir)




