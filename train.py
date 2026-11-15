import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
from src.data.dataset import ScalogramDataset
from src.models.vit import create_hf_vit_regression
from src.utils.metrics import my_loss_function, get_average_mae

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT for Heart Rate Regression")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to processed numpy files")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument('--max_iters', type=int, default=4000, help="Max iterations per epoch")
    parser.add_argument('--val_iters', type=int, default=100, help="Max iterations per validation phase")
    parser.add_argument('--num_workers', type=int, default=2, help="Dataloader workers")
    parser.add_argument('--save_dir', type=str, default="./checkpoints", help="Directory to save checkpoints")
   
    # Subject lists based on notebook output
    parser.add_argument('--train_subs', nargs='+', default=['S1', 'S2', 'S4', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'])
    parser.add_argument('--val_subs', nargs='+', default=['S3', 'S5'])
    parser.add_argument('--test_subs', nargs='+', default=['S6', 'S14', 'S15'])
   
    return parser.parse_args()

def main():
    args = parse_args()
   
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate DataLoaders
    train_ds = ScalogramDataset(args.train_subs, args.data_dir)
    val_ds = ScalogramDataset(args.val_subs, args.data_dir)
    test_ds = ScalogramDataset(args.test_subs, args.data_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. Setup Model
    model = create_hf_vit_regression().to(device)
    print(f"Hugging Face ViT loaded for 4-channel Heart Rate Regression")

    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type)

    train_losses_history = []
    val_losses_history = []

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
       
        for i, (images, labels) in enumerate(train_loader):
            if i >= args.max_iters:
                break

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type):
                outputs = model(images).logits
                loss = my_loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses_history.append(loss.item())

            if i % 50 == 0:
                print(f"Train Epoch {epoch+1}, Iter {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        print("Running Validation...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if i >= args.val_iters:
                    break

                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()

                with torch.autocast(device_type=device.type):
                    outputs = model(images).logits
                    v_loss = my_loss_function(outputs, labels)

                val_losses_history.append(v_loss.item())

        # Save Checkpoint
        ckpt_path = os.path.join(args.save_dir, f'vit_ppg_checkpoint_e{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses_history,
            'val_loss': val_losses_history,
        }, ckpt_path)
        print(f"Epoch {epoch+1} complete. Checkpoint saved to {ckpt_path}")

    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    print(f'Train MAE: {get_average_mae(model, train_loader, device):.2f} BPM')
    print(f'Validation MAE: {get_average_mae(model, val_loader, device):.2f} BPM')
    print(f'Test MAE: {get_average_mae(model, test_loader, device):.2f} BPM')

    # Save training plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_history, label='Train MAE')
    plt.plot(val_losses_history, label='Validation MAE', alpha=0.7)
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (BPM)')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'loss_plot.png'))
    print(f"Loss plot saved to {args.save_dir}/loss_plot.png")

if __name__ == "__main__":
    main()