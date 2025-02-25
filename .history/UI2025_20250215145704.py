import sys
import os
import threading
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import config
from hotpicture import load_model


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

        # 训练轮次设置
        self.epochs_label = QtWidgets.QLabel("训练轮次:")
        self.epochs_input = QtWidgets.QSpinBox()
        self.epochs_input.setMinimum(1)
        self.epochs_input.setValue(25)  # 默认25轮
        layout.addWidget(self.epochs_label)
        layout.addWidget(self.epochs_input)

        # 训练按钮
        self.train_button = QtWidgets.QPushButton("开始训练")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # 停止训练按钮
        self.stop_button = QtWidgets.QPushButton("停止训练")
        self.stop_button.clicked.connect(self.stop_training)
        layout.addWidget(self.stop_button)

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

    def train_model(self):
        if not config.train_datasetPatn:
            QMessageBox.critical(self, "错误", "请先选择训练数据集路径！")
            return

        # 获取训练轮次
        num_epochs = self.epochs_input.value()

        def train():
            self.model = load_model(config.model_path)
            self.log(f"训练开始，轮次：{num_epochs}")
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

    def log(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # 这里是将新日志信息添加到顶部
        self.log_text.setText(f"{timestamp} - {message}\n" + self.log_text.toPlainText())
        self.log_text.moveCursor(QtGui.QTextCursor.Start)  # 确保光标在最上方


# 启动应用
def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
