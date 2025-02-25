import sys
import os
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator


# 导入 matplotlib 嵌入式画布
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------
# 训练子线程：在子线程中运行训练任务，并通过信号传递日志和每个 epoch 的数据更新
# -------------------------
class TrainingWorker(QThread):
    # 发送日志消息信号
    log_signal = pyqtSignal(str)
    # 训练完成信号：(success, message)
    finished_signal = pyqtSignal(bool, str)
    # 每个 epoch 更新信号：(epoch, train_loss, train_acc, valid_loss, valid_acc)
    epoch_update_signal = pyqtSignal(int, float, float, float, float)

    def __init__(self, config_path, num_epochs, parent=None):
        super(TrainingWorker, self).__init__(parent)
        self.config_path = config_path
        self.num_epochs = num_epochs

    def run(self):
        try:
            self.log_signal.emit("开始加载配置...\n")
            # 动态加载用户选择的配置文件，模块名固定为 "config"
            spec = importlib.util.spec_from_file_location("config", self.config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            self.log_signal.emit(f"配置文件加载成功：{self.config_path}\n")
            
            # 导入 Train_Model 模块（确保与此 UI 在同一目录下）
            import Train_Model

            self.log_signal.emit("开始训练...\n")
            
            # 定义回调函数，每个 epoch 结束时调用，将训练数据发送给 UI 更新曲线
            def epoch_callback(epoch, train_loss, train_acc, valid_loss, valid_acc):
                self.epoch_update_signal.emit(epoch, train_loss, train_acc, valid_loss, valid_acc)
                self.log_signal.emit(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}\n"
                )
            
            # 调用 Train_Model.train_model 进行训练，传入用户指定的训练轮次和回调函数
            model = Train_Model.train_model(
                Train_Model.model, Train_Model.criterion, Train_Model.optimizer, Train_Model.scheduler,
                num_epochs=self.num_epochs,
                epoch_callback=epoch_callback
            )
            
            # 构建模型保存路径，并调用保存函数
            model_save_path = os.path.join(config.folder_name, 'trained_model.pth')
            Train_Model.save_model(model, model_save_path)
            
            self.log_signal.emit("训练完成！模型已保存到：" + model_save_path + "\n")
            self.finished_signal.emit(True, "训练完成")
        except Exception as e:
            self.log_signal.emit("训练过程中出现错误：" + str(e) + "\n")
            self.finished_signal.emit(False, str(e))

# -------------------------
# Matplotlib 画布，用于显示动态曲线（Loss 与 Accuracy）
# -------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 创建两个子图：左边显示 Loss，右边显示 Accuracy
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

# -------------------------
# 主窗口 UI
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("训练 UI")
        self.resize(900, 700)

        self.config_path = None

        # 配置文件选择按钮
        self.selectConfigBtn = QPushButton("选择配置文件")
        self.selectConfigBtn.clicked.connect(self.select_config_file)

        # 标签和文本框：用于输入训练轮次
        self.epochLabel = QLabel("训练轮次:")
        self.epochLineEdit = QLineEdit()
        self.epochLineEdit.setValidator(QIntValidator(1, 10000, self))
        self.epochLineEdit.setText("50")  # 默认训练 50 轮

        # 开始训练按钮
        self.trainBtn = QPushButton("开始训练")
        self.trainBtn.clicked.connect(self.start_training)

        # 将配置选择、训练轮次输入和训练按钮放在一行
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.selectConfigBtn)
        top_layout.addWidget(self.epochLabel)
        top_layout.addWidget(self.epochLineEdit)
        top_layout.addWidget(self.trainBtn)

        # 日志文本框，用于显示训练日志
        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setFont(QFont("Courier", 10))

        # 动态曲线绘图区域：嵌入 Matplotlib 画布，用于实时更新 Loss 与 Accuracy 曲线
        self.canvas = MplCanvas(self, width=8, height=3, dpi=100)
        # 初始化动态曲线数据列表
        self.epochs = []
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

        # 主布局，将上部按钮和绘图区域、下部日志区域组合在一起
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.logText)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 保存训练按钮初始样式
        self.trainBtn_default_text = self.trainBtn.text()

    def select_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "Python Files (*.py)")
        if file_path:
            self.config_path = file_path
            self.log("已选择配置文件: " + file_path + "\n")

    def log(self, message):
        self.logText.append(message)
        self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())

    def start_training(self):
        if not self.config_path:
            self.log("请先选择配置文件！\n")
            return

        try:
            num_epochs = int(self.epochLineEdit.text())
        except ValueError:
            self.log("请输入有效的训练轮次！\n")
            return

        # 更新训练按钮状态，改变样式表示训练开始
        self.trainBtn.setEnabled(False)
        self.trainBtn.setText("训练中...")

        # 清空之前动态曲线数据
        self.epochs.clear()
        self.train_losses.clear()
        self.valid_losses.clear()
        self.train_accs.clear()
        self.valid_accs.clear()
        self.update_plot()

        # 启动训练线程，将用户输入的训练轮次传递进去
        self.worker = TrainingWorker(self.config_path, num_epochs)
        self.worker.log_signal.connect(self.log)
        self.worker.epoch_update_signal.connect(self.update_epoch_data)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

    def update_epoch_data(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.train_accs.append(train_acc)
        self.valid_accs.append(valid_acc)
        self.update_plot()

    def update_plot(self):
        # 更新 Loss 曲线
        self.canvas.ax_loss.clear()
        self.canvas.ax_loss.plot(self.epochs, self.train_losses, label="Train Loss", color="blue")
        self.canvas.ax_loss.plot(self.epochs, self.valid_losses, label="Valid Loss", color="red")
        self.canvas.ax_loss.set_title("Loss")
        self.canvas.ax_loss.set_xlabel("Epoch")
        self.canvas.ax_loss.set_ylabel("Loss")
        self.canvas.ax_loss.legend()
        
        # 更新 Accuracy 曲线
        self.canvas.ax_acc.clear()
        self.canvas.ax_acc.plot(self.epochs, self.train_accs, label="Train Acc", color="green")
        self.canvas.ax_acc.plot(self.epochs, self.valid_accs, label="Valid Acc", color="orange")
        self.canvas.ax_acc.set_title("Accuracy")
        self.canvas.ax_acc.set_xlabel("Epoch")
        self.canvas.ax_acc.set_ylabel("Accuracy")
        self.canvas.ax_acc.legend()
        
        self.canvas.draw()

    def training_finished(self, success, message):
        # 训练结束后恢复训练按钮状态
        self.trainBtn.setEnabled(True)
        self.trainBtn.setText(self.trainBtn_default_text)
        self.log("训练结束: " + message + "\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
