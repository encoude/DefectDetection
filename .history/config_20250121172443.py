# 设置训练使用的设备
device = 'cuda' #'cpu'

# 训练结果保存的文件夹名称
folder_name = "Defect1"

# 设置模型路径和输出文件夹
model_path = r'Defect\trained_model.pth'
output_dir = r'images_Defect'

#训练集和验证集路径
train_datasetPatn = r'Data\leather\train'
valid_datasetPatn = r'Data\leather\val'

#热力图颜色值叠加值
alpha = 0.6