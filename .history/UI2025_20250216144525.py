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

# 推理窗口
class InferenceUI(QWidget):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("推理窗口")
        layout = QVBoxLayout()

        self.image_label = QLabel("选择要推理的图片：")
        layout.addWidget(self.image_label)

        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_button)

        self.result_label = QLabel("推理结果：")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.bmp *.png *.jpg *.jpeg)")
        if file_path:
            result = inference(self.model, file_path, self.device)
            self.result_label.setText(f"推理结果：{result}")


# 训练界面 UI
class TrainingUI(QWidget):
    trainLogSignal = pyqtSignal(str)
    userLogSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainLogSignal.connect(self.append_train_log)
        self.userLogSignal.connect(self.append_user_log)

    def init_ui(self):
        self.setWindowTitle("缺陷检测模型训练")
        
        # 设置窗口样式，避免样式冲突
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        # 启动全屏并强制刷新布局
        self.showMaximized()
        QApplication.processEvents()

        # 顶部输入与按钮区域
        self.config_label = QLabel("选择配置文件：")
        self.config_button = QPushButton("选择文件")
        self.config_button.clicked.connect(self.select_config_file)

        self.epoch_label = QLabel("输入训练轮次：")
        self.epoch_input = QLineEdit()
        self.epoch_input.setText("25")

        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)

        self.inference_button = QPushButton("推理")
        self.inference_button.clicked.connect(self.open_inference_window)

        # 布局设置
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.config_label)
        top_layout.addWidget(self.config_button)
        top_layout.addWidget(self.epoch_label)
        top_layout.addWidget(self.epoch_input)
        top_layout.addWidget(self.train_button)
        top_layout.addWidget(self.inference_button)

        # Matplotlib 画布
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)

        # 日志框
        self.log_text_train = QTextEdit()
        self.log_text_user = QTextEdit()

        # 布局组合
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.log_text_train)
        layout.addWidget(self.log_text_user)

        self.setLayout(layout)

    def select_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "Python Files (*.py)")
        if file_path:
            self.userLogSignal.emit(f"已选择配置文件：{file_path}")

    def append_train_log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.log_text_train.append(f"{timestamp} - {message}")

    def append_user_log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.log_text_user.append(f"{timestamp} - {message}")

    def start_training(self):
        num_epochs = int(self.epoch_input.text())
        self.userLogSignal.emit(f"训练轮次设置为: {num_epochs}轮")

        # 这里需要添加你的训练逻辑

    def open_inference_window(self):
        self.userLogSignal.emit("打开推理窗口")
        self.inference_window = InferenceUI(None, self.device)
        self.inference_window.show()

# 启动应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TrainingUI()
    ui.show()
    sys.exit(app.exec_())
