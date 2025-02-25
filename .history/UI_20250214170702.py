import sys
import os
import torch
import configparser
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
                             QFileDialog, QLineEdit, QTextEdit, QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import Train_Model

class TrainingUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('训练模型界面')
        self.setGeometry(100, 100, 800, 600)

        # 配置文件选择
        self.config_label = QLabel('配置文件路径:')
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setReadOnly(True)
        self.config_btn = QPushButton('选择配置文件')
        self.config_btn.clicked.connect(self.select_config_file)

        # 训练轮次输入
        self.epochs_label = QLabel('训练轮次:')
        self.epochs_input = QLineEdit()
        self.epochs_input.setPlaceholderText('请输入训练轮次')

        # 启动训练按钮
        self.train_btn = QPushButton('开始训练')
        self.train_btn.clicked.connect(self.start_training)

        # 日志输出文本框
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Matplotlib 图表
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax_loss = self.figure.add_subplot(121)
        self.ax_acc = self.figure.add_subplot(122)

        # 布局
        layout = QVBoxLayout()

        # 配置文件选择部分
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_path_edit)
        config_layout.addWidget(self.config_btn)
        layout.addLayout(config_layout)

        # 训练轮次输入部分
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(self.epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        layout.addLayout(epochs_layout)

        # 启动按钮
        layout.addWidget(self.train_btn)

        # 日志输出
        layout.addWidget(self.log_output)

        # 动态曲线
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # 初始化变量
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

    def select_config_file(self):
        """选择配置文件"""
        config_file, _ = QFileDialog.getOpenFileName(self, '选择配置文件', '', 'Python Files (*.py)')
        if config_file:
            self.config_path_edit.setText(config_file)

    def log(self, message):
        """在日志框中输出消息"""
        self.log_output.append(message)

    def epoch_callback(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        """每个 epoch 结束后更新日志和曲线"""
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.train_accuracies.append(train_acc)
        self.valid_accuracies.append(valid_acc)

        # 更新日志
        self.log(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, ' 
                 f'Valid Loss = {valid_loss:.4f}, Valid Accuracy = {valid_acc:.4f}')

        # 更新曲线
        self.ax_loss.clear()
        self.ax_acc.clear()

        self.ax_loss.plot(self.train_losses, label='Train Loss')
        self.ax_loss.plot(self.valid_losses, label='Valid Loss')
        self.ax_loss.set_title('Loss per Epoch')
        self.ax_loss.legend()

        self.ax_acc.plot(self.train_accuracies, label='Train Accuracy')
        self.ax_acc.plot(self.valid_accuracies, label='Valid Accuracy')
        self.ax_acc.set_title('Accuracy per Epoch')
        self.ax_acc.legend()

        self.canvas.draw()

    def start_training(self):
        """开始训练模型"""
        config_path = self.config_path_edit.text()
        if not config_path:
            self.log('请先选择配置文件！')
            return

        epochs = self.epochs_input.text()
        if not epochs.isdigit():
            self.log('请输入有效的训练轮次！')
            return

        # 加载配置文件
        self.log('加载配置文件...')
        sys.modules['config'] = __import__(config_path.replace('.py', ''))

        # 禁用按钮，防止重复点击
        self.train_btn.setEnabled(False)
        self.train_btn.setText('训练中...')

        # 开始训练
        def callback(epoch, train_loss, train_acc, valid_loss, valid_acc):
            self.epoch_callback(epoch, train_loss, train_acc, valid_loss, valid_acc)

        model = Train_Model.train_model(Train_Model.model, Train_Model.criterion, Train_Model.optimizer, 
                                        Train_Model.scheduler, num_epochs=int(epochs), callback=callback)

        # 训练完成
        self.train_btn.setEnabled(True)
        self.train_btn.setText('开始训练')
        self.log('训练完成！')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainingUI()
    window.show()
    sys.exit(app.exec_())
