import torch
from torchvision.models import ResNet18_Weights
  # 替换为您的模型类

# 初始化模型
model = ResNet18_Weights()
# 加载权重
model.load_state_dict(torch.load(r'F:\Python\AI\DefectDetection\2025.1.5\Defect\trained_model.pth'))
model.eval()  # 设置为评估模式

# 假设模型期望的输入是一个形状为 (1, 3, 224, 224) 的张量
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,               # 要转换的模型
    dummy_input,         # 示例输入
    'model.onnx',        # 输出文件名
    input_names=['input'],   # 输入名称
    output_names=['output'], # 输出名称
    opset_version=11,        # ONNX 算子集版本
    do_constant_folding=True # 是否执行常量折叠优化
)

import onnx

# 加载 ONNX 模型
onnx_model = onnx.load('model.onnx')
# 检查模型
onnx.checker.check_model(onnx_model)
print('ONNX 模型已成功验证。')
