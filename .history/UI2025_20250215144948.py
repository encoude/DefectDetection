import sys
import os
import threading
import time
import torch
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import config
from hotpicture import detect_and_save, load_model


class MainApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口
        self.setWindowTitle("Defect Detection GUI")
        self.setGeometry(100, 100, 800, 600)

        # 初始化UI
        self.init_ui()

        # 定义模型变量
        self.model = None
        self.training_thread = None

    def init_ui(self):
        # 布局
        layout = QtWidgets.QVBoxLayout()

        # 配置文件选择按钮
        self.config_button = QtWidgets.QPushButton("选择配置文件")
        self.config_button.clicked.connect(self.load_config)
        layout.addWidget(self.config_button)

        # 训练图片路径按钮
        self.train_image_button = QtWidgets.QPushButton("选择训练图片路径")
        self.train_image_button.clicked.connect(self.load_train_images)
        layout.addWidget(self.train_image_button)

        # 训练按钮
        self.train_button = QtWidgets.QPushButton("开始训练")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # 停止训练按钮
        self.stop_button = QtWidgets.QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_button)

        # 模型预测按钮
        self.predict_button = QtWidgets.QPushButton("进行预测")
        self.predict_button.clicked.connect(self.predict_model)
        layout.addWidget(self.predict_button)

        # 日志窗口
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 创建图形区域
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.ax[0].set_title('Loss per Epoch')
        self.ax[1].set_title('Accuracy per Epoch')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # 设置布局
        self.setLayout(layout)

    def load_config(self):
        config_file, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "Python Files (*.py)")
        if config_file:
            # 这里可以实现加载配置文件的功能
            QMessageBox.information(self, "信息", "配置文件已加载！")
            self.log("配置文件加载成功")

    def load_train_images(self):
        train_folder = QFileDialog.getExistingDirectory(self, "选择训练图片文件夹")
        if train_folder:
            config.train_datasetPatn = train_folder
            QMessageBox.information(self, "信息", "训练图片路径已加载！")
            self.log(f"训练数据集路径设置为：{train_folder}")

    def train_model(self):
        if not config.train_datasetPatn:
            QMessageBox.critical(self, "错误", "请先选择训练数据集路径！")
            return

        def train():
            self.model = load_model(config.model_path)
            self.log("训练开始...")
            # 调用实际的训练方法
            # 假设训练过程用 `train_model` 进行并输出损失、准确率
            self.log("训练完成")
            QMessageBox.information(self, "信息", "训练完成！")

        self.training_thread = threading.Thread(target=train)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log("训练已停止")
            # 这里实现停止训练的逻辑
        else:
            self.log("没有正在进行的训练任务")

    def predict_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pth)")
        if model_file:
            self.model = load_model(model_file)
            image_file, _ = QFileDialog.getOpenFileName(self, "选择验证图片", "", "Image Files (*.bmp;*.jpg;*.png)")
            if image_file:
                self.predict(image_file)

    def predict(self, image_path):
        if self.model is None:
            QMessageBox.critical(self, "错误", "没有加载模型！")
            return

        # 执行预测
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if output_dir:
            detect_and_save(self.model, image_path, output_dir)
            self.log(f"预测完成，结果保存至：{output_dir}")

    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.log_text.append(f"{timestamp} - {message}")
        self.log_text.moveCursor(QtGui.QTextCursor.End)  # 滚动到最新日志

# 启动应用
def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
