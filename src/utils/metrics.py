import torch
import torch.nn as nn

criterion = nn.L1Loss()

def my_loss_function(predictions, targets):
    """
    Calculates the Mean Absolute Error between predicted and actual heart rate.
    Expects model_output.logits and targets to be same shape [Batch, 1]
    """
    return criterion(predictions, targets)

def get_average_mae(model, data_loader, device):
    model.eval()
    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(images).logits
            error = torch.abs(outputs - labels).sum().item()

            total_error += error
            total_samples += labels.size(0)

    if total_samples == 0:
        return 0.0
    return total_error / total_samples