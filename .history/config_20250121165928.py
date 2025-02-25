# 设置训练使用的设备
device = 'cuda' #'cpu'

# 训练结果保存的文件夹名称
folder_name = "Defect1"

# 设置模型路径和输出文件夹
model_path = r'Defect1\trained_model.pth'
output_dir = r'Defect_images'


#训练集和验证集路径
train_datasetPatn = r'Data\marbled\train'
valid_datasetPatn = r'Data\marbled\val'