import torch
import matplotlib.pyplot as plt

def plot_hr_time_series(model, data_loader, device, num_windows=100, save_path=None):
    """
    Evaluates the model on the data_loader and plots the predicted vs actual HR.
    """
    model.eval()
    predictions = []
    actuals = []
   
    with torch.no_grad():
        for images, labels in data_loader:
            if len(predictions) >= num_windows:
                break # Stop after enough samples are collected
           
            images = images.to(device)
            outputs = model(images).logits
           
            # Extract lists safely
            preds = outputs.cpu().squeeze().tolist()
            if not isinstance(preds, list):
                preds = [preds] # Handle edge case where batch_size=1
               
            predictions.extend(preds)
            actuals.extend(labels.tolist())

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(actuals[:num_windows], label='Ground Truth (BPM)', color='black', linewidth=2)
    plt.plot(predictions[:num_windows], label='ViT Prediction (BPM)', color='red', linestyle='--')
    plt.title('Heart Rate Estimation Over Time')
    plt.xlabel('Window Index')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
   
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()