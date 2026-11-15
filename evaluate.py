import argparse
import torch
import os

# from src.data.dataset import get_dataloaders
from torch.utils.data import DataLoader
from src.data.dataset import ScalogramDataset
from src.models.vit import create_hf_vit_regression
from src.utils.visualize import plot_hr_time_series

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and plot ViT Heart Rate Predictions")
   
    # Data & Model args
    parser.add_argument('--data_dir', type=str, default='./Processed_NP', help='Path to processed dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved .pth model checkpoint')

    parser.add_argument('--val_subs', nargs='+', default=['S3', 'S5'])
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--num_workers', type=int, default=2, help="Dataloader workers")
   
    # Plotting args
    parser.add_argument('--num_windows', type=int, default=100, help='Number of time windows to plot')
    parser.add_argument('--save_path', type=str, default='./hr_evaluation_plot.png', help='File path to save the plot')
   
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Validation Data
    print(f"Loading data from {args.data_dir}...")
    # We only need the validation loader for this script, but we reuse your existing function
    # _, val_loader, _ = get_dataloaders(
    #     data_dir=args.data_dir,
    #     batch_size=32,
    #     num_workers=2
    # )
    val_ds = ScalogramDataset(args.val_subs, args.data_dir)

    val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)


    # 2. Initialize Model
    print("Initializing Hugging Face ViT...")
    model = create_hf_vit_regression().to(device)

    # 3. Load Checkpoint Weights
    print(f"Loading weights from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
       
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Weights loaded successfully.")

    # 4. Generate Plot
    print(f"Generating plot for {args.num_windows} windows...")
    plot_hr_time_series(
        model=model,
        data_loader=val_loader,
        device=device,
        num_windows=args.num_windows,
        save_path=args.save_path
    )

if __name__ == '__main__':
    main()