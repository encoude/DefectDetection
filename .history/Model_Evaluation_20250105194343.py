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

# 评估模型在验证集上的表现
evaluate_model(r'F:\python\AI\DefectDetection\2025.1.5\best_model.pth', valid_loader)
