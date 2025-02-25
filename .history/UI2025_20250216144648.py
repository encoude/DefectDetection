import os
import time
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
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QWidget, QLabel, QFileDialog, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal

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
        
        # 保持原有布局，不改动布局结构
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.showMaximized()
        QApplication.processEvents()  # 确保界面正确刷新，避免缩放异常

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

        # 日志框
        self.log_text_train = QTextEdit()
        self.log_text_user = QTextEdit()

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
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
        # 在此处添加你的训练逻辑

    def open_inference_window(self):
        self.userLogSignal.emit("打开推理窗口")
        # 省略推理窗口的实现
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TrainingUI()
    ui.show()
    sys.exit(app.exec_())
