import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import torch
from PIL import Image
import config
import numpy as np
from hotpicture import detect_and_save, load_model

# 主应用窗口
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Defect Detection GUI")
        self.root.geometry("800x600")

        # 配置文件按钮
        self.config_button = tk.Button(root, text="选择配置文件", command=self.load_config)
        self.config_button.pack(pady=10)

        # 训练图片路径按钮
        self.train_image_button = tk.Button(root, text="选择训练图片路径", command=self.load_train_images)
        self.train_image_button.pack(pady=10)

        # 训练按钮
        self.train_button = tk.Button(root, text="开始训练", command=self.train_model)
        self.train_button.pack(pady=10)

        # 停止训练按钮
        self.stop_button = tk.Button(root, text="停止训练", command=self.stop_training)
        self.stop_button.pack(pady=10)

        # 模型预测按钮
        self.predict_button = tk.Button(root, text="进行预测", command=self.predict_model)
        self.predict_button.pack(pady=10)

        # 日志窗口
        self.log_text = tk.Text(root, height=10, width=80)
        self.log_text.pack(pady=10)
        self.log_text.insert(tk.END, "日志信息会显示在这里...\n")

        # 图形区域
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.ax[0].set_title('Loss per Epoch')
        self.ax[1].set_title('Accuracy per Epoch')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

        # 其他变量
        self.model = None
        self.training_thread = None

    def load_config(self):
        config_file = filedialog.askopenfilename(title="选择配置文件", filetypes=[("Text Files", "*.py")])
        if config_file:
            # 这里可以实现加载配置文件的功能
            messagebox.showinfo("信息", "配置文件已加载！")
            self.log("配置文件加载成功")

    def load_train_images(self):
        train_folder = filedialog.askdirectory(title="选择训练图片文件夹")
        if train_folder:
            config.train_datasetPatn = train_folder
            messagebox.showinfo("信息", "训练图片路径已加载！")
            self.log(f"训练数据集路径设置为：{train_folder}")

    def train_model(self):
        if not config.train_datasetPatn:
            messagebox.showerror("错误", "请先选择训练数据集路径！")
            return

        def train():
            self.model = load_model(config.model_path)
            self.log("训练开始...")
            # 这里可以调用实际的训练方法
            # 更新实时图形和日志
            # 在训练期间绘制损失和准确率曲线
            # 假设训练过程用 `train_model` 进行并输出损失、准确率
            self.log("训练完成")
            messagebox.showinfo("信息", "训练完成！")

        self.training_thread = threading.Thread(target=train)
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log("训练已停止")
            # 这里实现停止训练的逻辑
        else:
            self.log("没有正在进行的训练任务")

    def predict_model(self):
        model_file = filedialog.askopenfilename(title="选择模型文件", filetypes=[("Model Files", "*.pth")])
        if model_file:
            self.model = load_model(model_file)
            image_file = filedialog.askopenfilename(title="选择验证图片", filetypes=[("Image Files", "*.bmp;*.jpg;*.png")])
            if image_file:
                self.predict(image_file)

    def predict(self, image_path):
        if self.model is None:
            messagebox.showerror("错误", "没有加载模型！")
            return

        # 执行预测
        output_dir = filedialog.askdirectory(title="选择输出文件夹")
        if output_dir:
            detect_and_save(self.model, image_path, output_dir)
            self.log(f"预测完成，结果保存至：{output_dir}")

    def log(self, message):
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.log_text.yview(tk.END)  # 自动滚动到最新日志

# 启动GUI
def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
