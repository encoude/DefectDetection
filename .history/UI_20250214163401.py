import sys
import os
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, QPropertyAnimation, QPoint

# 训练子线程
class TrainingWorker(QThread):
    # 发出日志信号
    log_signal = pyqtSignal(str)
    # 发出训练结束信号：success表示是否成功，message为描述信息
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config_path, parent=None):
        super(TrainingWorker, self).__init__(parent)
        self.config_path = config_path

    def run(self):
        try:
            self.log_signal.emit("开始加载配置...\n")
            # 动态加载选择的配置文件，模块名称固定为 "config"
            spec = importlib.util.spec_from_file_location("config", self.config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            self.log_signal.emit(f"配置文件加载成功：{self.config_path}\n")
            
            # 导入 Train_Model 模块（确保在同一目录下）
            import Train_Model

            self.log_signal.emit("开始训练...\n")
            # 调用 Train_Model.train_model 进行训练，这里假设 Train_Model.py 中已定义好 model, criterion, optimizer, scheduler
            # 训练过程中会打印日志（你可以在 Train_Model.train_model 内部添加 print 或日志输出）
            model = Train_Model.train_model(Train_Model.model, Train_Model.criterion, 
                                            Train_Model.optimizer, Train_Model.scheduler, num_epochs=200)
            
            # 构建模型保存路径，并调用保存函数
            model_save_path = os.path.join(config.folder_name, 'trained_model.pth')
            Train_Model.save_model(model, model_save_path)
            
            self.log_signal.emit("训练完成！模型已保存到：" + model_save_path + "\n")
            self.finished_signal.emit(True, "训练完成")
        except Exception as e:
            self.log_signal.emit("训练过程中出现错误：" + str(e) + "\n")
            self.finished_signal.emit(False, str(e))

# 主窗口 UI
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("训练 UI")
        self.resize(600, 400)

        self.config_path = None
        self.original_train_btn_pos = None

        # 创建选择配置文件按钮
        self.selectConfigBtn = QPushButton("选择配置文件")
        self.selectConfigBtn.clicked.connect(self.select_config_file)

        # 创建开始训练按钮
        self.trainBtn = QPushButton("开始训练")
        self.trainBtn.clicked.connect(self.start_training)

        # 布局：将两个按钮放在一行
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.selectConfigBtn)
        btn_layout.addWidget(self.trainBtn)

        # 创建日志显示文本框
        self.logText = QTextEdit()
        self.logText.setReadOnly(True)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.logText)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 记录开始训练按钮的原始位置，用于动画
        self.original_train_btn_pos = self.trainBtn.pos()

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

        self.trainBtn.setEnabled(False)
        # 开始训练前，动画将训练按钮向右滑动（例如向右滑动 100 像素）
        self.animate_button(self.trainBtn, self.trainBtn.pos(), self.trainBtn.pos() + QPoint(100, 0))
        
        # 启动训练线程
        self.worker = TrainingWorker(self.config_path)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

    def training_finished(self, success, message):
        # 训练完成或出现问题时，将按钮动画滑回原位
        self.animate_button(self.trainBtn, self.trainBtn.pos(), self.original_train_btn_pos)
        self.trainBtn.setEnabled(True)

    def animate_button(self, button, start_pos, end_pos):
        animation = QPropertyAnimation(button, b"pos")
        animation.setDuration(500)  # 动画持续时间 500 毫秒
        animation.setStartValue(start_pos)
        animation.setEndValue(end_pos)
        animation.start()
        # 防止动画对象被垃圾回收，保存引用
        self.current_animation = animation

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
