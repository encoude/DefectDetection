import os
import time
import config
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import colormaps
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import threading
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QWidget, QLabel, QFileDialog, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QTextCursor, QFont


# 设备类型
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

# 训练模型函数
def train_model(ui, model, criterion, optimizer, scheduler, num_epochs=25):
    ui.trainLogSignal.emit("开始训练...")
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    for epoch in range(num_epochs):
        ui.trainLogSignal.emit(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            dataloader = ui.train_loader if phase == 'train' else ui.valid_loader
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(ui.device), labels.to(ui.device)
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
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                valid_losses.append(epoch_loss)
                valid_accuracies.append(epoch_acc.item())

            ui.trainLogSignal.emit(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            # 更新图表
            ui.update_plot(train_losses, valid_losses, train_accuracies, valid_accuracies)
        scheduler.step()
    ui.trainLogSignal.emit("训练完成！")
    return model

# 加载模型
def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 修改为 weights_only=True
    return model.to(device)

#修改最后卷积层的提取方式
def get_final_conv_layer(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            final_conv_layer = module
    return final_conv_layer

# Grad-CAM 实现
def grad_cam(model, img_tensor):
    model.train()  # 确保梯度计算
    img_tensor = img_tensor.requires_grad_(True)

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    final_conv_layer = get_final_conv_layer(model)
    if final_conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")
    
    gradient = None
    activation_map = None
    
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0]  # 提取梯度
    
    def save_activation_map(module, input, output):
        nonlocal activation_map
        activation_map = output  # 提取激活图
    
    hook_activations = final_conv_layer.register_forward_hook(save_activation_map)
    hook_gradients = final_conv_layer.register_full_backward_hook(save_gradient)  # 修改为 register_full_backward_hook
    
    with torch.enable_grad():
        output = model(img_tensor)
        target_class = output.argmax(dim=1)
        target = output[0, target_class]

    model.zero_grad()
    target.backward(retain_graph=True)
    
    if activation_map is None or gradient is None:
        print("Error: Activations or gradients are not captured.")
        return None

    gradients = gradient[0]
    activations = activation_map[0]
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * activations, dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().detach().numpy()
    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (224, 224))

    hook_activations.remove()
    hook_gradients.remove()

    return cam

def apply_jet_colormap(heatmap):
    normalized_heatmap = heatmap / 255.0
    jet_colormap = colormaps.get_cmap('jet')  # 使用 jet colormap
    colored_heatmap = jet_colormap(normalized_heatmap)
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    return colored_heatmap

# 将热力图与原始图像叠加
def overlay_heatmap_with_alpha(img, heatmap):
    img = np.array(img)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    jet_heatmap = apply_jet_colormap(heatmap)
    overlay = cv2.addWeighted(img, 1 - config.alpha, jet_heatmap, config.alpha, 0)
    return overlay

class InferenceUI(QWidget):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("推理窗口")
        self.setWindowState(Qt.WindowMaximized)  # 设置为最大化窗口
        layout = QVBoxLayout()

        self.select_model_button = QPushButton("选择模型")
        self.select_model_button.clicked.connect(self.select_model)
        layout.addWidget(self.select_model_button)

        self.image_label = QLabel("选择要推理的图片：")
        layout.addWidget(self.image_label)

        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        # 创建显示图的区域
        self.image_display_layout = QHBoxLayout()
        self.original_image_label = QLabel("原始图像")
        self.heatmap_image_label = QLabel("热力图")
        self.image_display_layout.addWidget(self.original_image_label)
        self.image_display_layout.addWidget(self.heatmap_image_label)

        layout.addLayout(self.image_display_layout)

        # 日志区域
        self.log_label = QLabel("推理日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_label)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PyTorch Models (*.pth *.pt)")
        if file_path:
            self.model = load_model(file_path)
            self.log_text.append(f"已加载模型: {file_path}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.bmp *.png *.jpg *.jpeg)")
        if file_path:
            img = Image.open(file_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            heatmap = grad_cam(self.model, img_tensor)
            overlayed_img = overlay_heatmap_with_alpha(img, heatmap)

            # 显示原始图像和热力图
            self.original_image_label.setPixmap(self.image_to_pixmap(img))
            self.heatmap_image_label.setPixmap(self.image_to_pixmap(Image.fromarray(overlayed_img)))

            self.log_text.append(f"推理结果: {'缺陷' if heatmap is not None else '无缺陷'}")

    def image_to_pixmap(self, img):
        img = img.convert("RGB")
        img = img.toqpixmap()
        return img


# 训练界面 UI
class TrainingUI(QWidget):
    # 定义信号，用于跨线程更新日志
    trainLogSignal = pyqtSignal(str)
    userLogSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 连接信号与槽，确保日志更新在主线程中执行
        self.trainLogSignal.connect(self.append_train_log)
        self.userLogSignal.connect(self.append_user_log)

    def init_ui(self):
        self.setWindowTitle("缺陷检测模型训练")
        self.setWindowState(Qt.WindowMaximized)  # 确保窗口最大化

        # 配置文件选择
        self.config_label = QLabel("选择配置文件：")
        self.config_button = QPushButton("选择文件")
        self.config_button.setStyleSheet(""" 
        QPushButton {
            background-color: #6A5ACD;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #397D34;
        }
    """)
        self.config_button.clicked.connect(self.select_config_file)

        # 训练轮次输入
        self.epoch_label = QLabel("输入训练轮次：")
        self.epoch_input = QLineEdit()
        self.epoch_input.setStyleSheet(""" 
        QLineEdit {
            background-color: #F0F0F0;
            color: #333;
            font-size: 16px;
            padding: 8px;
            border: 2px solid #ccc;
            border-radius: 10px;
        }
        QLineEdit:focus {
            border: 2px solid #4CAF50;
            background-color: #FFFFFF;
        }
        QLineEdit:hover {
            border: 2px solid #66BB6A;
        }
    """)
        self.epoch_input.setText("25")

        # 用户操作日志
        self.log_text_user = QTextEdit()
        self.log_text_user.setReadOnly(True)

        # 训练按钮
        self.train_button = QPushButton("开始训练")
        self.train_button.setStyleSheet(""" 
        QPushButton {
            background-color: #6A5ACD;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #397D34;
        }
    """)
        self.train_button.clicked.connect(self.start_training)

        self.inference_button = QPushButton("推理")
        self.inference_button.clicked.connect(self.open_inference_window)

        # Matplotlib 画布
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)

        # 训练日志文本框
        self.log_text_train = QTextEdit()
        self.log_text_train.setReadOnly(True)

        # 设置标签字体
        label_font = QFont("Arial", 12, QFont.Bold)
        self.train_log_label = QLabel("训练日志")
        self.train_log_label.setFont(label_font)
        self.user_log_label = QLabel("用户操作日志")
        self.user_log_label.setFont(label_font)

        # 顶部区域布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.config_label)
        top_layout.addWidget(self.config_button)
        top_layout.addWidget(self.epoch_label)
        top_layout.addWidget(self.epoch_input)
        top_layout.addWidget(self.train_button)
        top_layout.addWidget(self.inference_button)


        # 日志框部分布局
        user_log_layout = QVBoxLayout()
        user_log_layout.addWidget(self.user_log_label)
        user_log_layout.addWidget(self.log_text_user)

        train_log_layout = QVBoxLayout()
        train_log_layout.addWidget(self.train_log_label)
        train_log_layout.addWidget(self.log_text_train)

        logs_layout = QHBoxLayout()
        logs_layout.addLayout(user_log_layout)
        logs_layout.addLayout(train_log_layout)

        # 主要区域布局
        layout = QVBoxLayout()
        layout.addLayout(top_layout)  # 顶部输入和按钮
        layout.addWidget(self.canvas, stretch=3)  # 图表部分占更多比例
        layout.addLayout(logs_layout, stretch=1)  # 日志部分

        self.setLayout(layout)


    def select_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "Python Files (*.py)")
        if file_path:
            self.userLogSignal.emit(f"已选择配置文件：{file_path}")

    def append_user_log(self, message):
        # 获取当前时间并附加到用户操作日志消息
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_message = f"{timestamp} - {message}"
        # 将新日志添加到文本框顶部
        self.log_text_user.setText(log_message + "\n" + self.log_text_user.toPlainText())
        self.log_text_user.moveCursor(QTextCursor.Start)

    def append_train_log(self, message):
        # 获取当前时间并附加到训练日志消息
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_message = f"{timestamp} - {message}"
        # 将新日志添加到文本框顶部
        self.log_text_train.setText(log_message + "\n" + self.log_text_train.toPlainText())
        self.log_text_train.moveCursor(QTextCursor.Start)

    def start_training(self):
        num_epochs = int(self.epoch_input.text())
        self.userLogSignal.emit(f"训练轮次设置为: {num_epochs}轮")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_dataset = DefectDataset(data_dir=config.train_datasetPatn, transform=transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.valid_dataset = DefectDataset(data_dir=config.valid_datasetPatn, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=32, shuffle=False)
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # 启动训练线程，避免阻塞 UI
        threading.Thread(target=train_model, args=(self, model, criterion, optimizer, scheduler, num_epochs)).start()

    def open_inference_window(self):
        self.userLogSignal.emit("打开推理窗口")
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(config.model_path, map_location=self.device))
        model = model.to(self.device)
        self.inference_window = InferenceUI(model, self.device)
        self.inference_window.show()

    def update_plot(self, train_losses, valid_losses, train_accuracies, valid_accuracies):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(train_losses, label='Train Loss')
        self.ax1.plot(valid_losses, label='Valid Loss')
        self.ax1.set_title('Loss per Epoch')
        self.ax1.legend()
        self.ax2.plot(train_accuracies, label='Train Accuracy')
        self.ax2.plot(valid_accuracies, label='Valid Accuracy')
        self.ax2.set_title('Accuracy per Epoch')
        self.ax2.legend()
        self.canvas.draw()

# 启动应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TrainingUI()
    ui.show()
    sys.exit(app.exec_())
