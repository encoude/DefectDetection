import os
import time
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
import sys
import threading
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QWidget, QLabel, QFileDialog, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QTextCursor, QFont

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
                if img_name.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(0 if label == 'no_defect' else 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

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

# UI 设计
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
        self.showMaximized()  # 启动时全屏显示

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
