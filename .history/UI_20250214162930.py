import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import os

# 导入训练配置和训练代码
import config
import Train_Model

# 训练过程在子线程中运行
def run_training():
    try:
        log("训练开始...\n")
        
        # 初始化模型、损失函数、优化器和学习率调度器（Train_Model.py 中已有相应代码）
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim import lr_scheduler
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights

        # 加载预训练 ResNet18 模型，并修改最后的全连接层
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 定义损失函数、优化器和学习率调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 调用 Train_Model.py 中的训练函数（这里训练 200 个 epoch，可根据需要修改）
        model = Train_Model.train_model(model, criterion, optimizer, scheduler, num_epochs=200)
        
        # 构建模型保存路径，并调用保存函数
        model_save_path = os.path.join(config.folder_name, 'trained_model.pth')
        Train_Model.save_model(model, model_save_path)
        
        log("训练完成。\n模型已保存到: " + model_save_path + "\n")
    except Exception as e:
        log("训练过程中出现错误： " + str(e) + "\n")
    finally:
        start_button.config(state=tk.NORMAL)

def start_training():
    start_button.config(state=tk.DISABLED)
    threading.Thread(target=run_training, daemon=True).start()

def log(message):
    text_area.insert(tk.END, message)
    text_area.see(tk.END)

# 创建主窗口
root = tk.Tk()
root.title("训练 UI 示例")

# 创建一个按钮框架
frame = tk.Frame(root)
frame.pack(pady=10)

# 开始训练按钮
start_button = tk.Button(frame, text="开始训练", width=15, command=start_training)
start_button.pack(side=tk.LEFT, padx=5)

# 退出按钮
exit_button = tk.Button(frame, text="退出", width=15, command=root.quit)
exit_button.pack(side=tk.LEFT, padx=5)

# 创建一个带滚动条的文本区域用于显示日志
text_area = scrolledtext.ScrolledText(root, width=80, height=20)
text_area.pack(padx=10, pady=10)

# 启动主事件循环
root.mainloop()
