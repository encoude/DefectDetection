# 设置训练使用的设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 训练结果保存的文件夹名称
folder_name = "training_results_folder"
output_dir = 'training_results'  # 文件夹名