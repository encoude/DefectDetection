def evaluate_model(model, dataloader):
    model.eval()
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Evaluation Accuracy: {accuracy:.4f}')

# 加载模型
model_path = r'F:\python\AI\DefectDetection\2025.1.5\best_model.pth'
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 初始化模型
model.fc = nn.Linear(model.fc.in_features, 2)  # 确保模型结构一致
model.load_state_dict(torch.load(model_path))  # 加载保存的权重
model = model.to(device)  # 将模型移至 GPU 或 CPU

# 验证集路径 (加载验证数据集)
valid_dataset = DefectDataset(data_dir=r"F:\项目软件\方昇\test3\val", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 评估模型在验证集上的表现
evaluate_model(model, valid_loader)
