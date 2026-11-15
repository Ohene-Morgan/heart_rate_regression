# Heart Rate Regression with Vision Transformer

Estimates heart rate (BPM) from PPG scalogram "images" using a Vision Transformer (ViT), by fine-tuning Google's pretrained `vit-base-patch16-224` on 4-channel input instead of the usual 3-channel RGB.

## How it works

- **Data**: Each sample is a 4-channel scalogram window (time-frequency representation of a PPG signal) paired with a ground-truth heart rate label, stored per-subject as `{subject}_X.npy` / `{subject}_y.npy`. `ScalogramDataset` (`src/data/dataset.py`) memory-maps these arrays so only the needed window is loaded into RAM per sample.
- **Model**: `create_hf_vit_regression` (`src/models/vit.py`) loads `google/vit-base-patch16-224` with its classification head swapped for a single regression output (`num_labels=1`), and its patch-embedding Conv2d expanded from 3 to 4 input channels (the 4th channel's weights are initialized from the pretrained red channel).
- **Loss / metric**: L1 loss (MAE) in BPM, implemented in `src/utils/metrics.py`.
- **Splits**: Subjects are split into train/val/test (configurable via CLI args), e.g. defaults: train `S1,S2,S4,S7-S13`, val `S3,S5`, test `S6,S14,S15`.

## Repo layout

```
train.py              # training loop, checkpointing, final MAE eval + loss plot
evaluate.py            # loads a checkpoint and plots predicted vs. actual HR over time
src/
  data/dataset.py      # ScalogramDataset
  models/vit.py         # 4-channel ViT regression model
  utils/metrics.py      # MAE loss + evaluation helper
  utils/visualize.py    # prediction time-series plotting
checkpoints/            # loss_plot.png (model .pth files are gitignored, ~1GB each)
processed_NP/           # expected location for {subject}_X.npy / {subject}_y.npy (gitignored, ~25GB)
hr_evaluation_plot_e*.png  # predicted vs. actual HR plots per epoch's checkpoint
log.txt                 # example training run log
```

Large artifacts (`checkpoints/*.pth`, `processed_NP/*.npy`) are excluded from version control — see `.gitignore`.

## Usage

```bash
pip install -r requirements.txt

# Train
python train.py --data_dir processed_NP --epochs 10 --batch_size 32

# Evaluate a checkpoint and plot predictions
python evaluate.py --data_dir processed_NP --checkpoint checkpoints/vit_ppg_checkpoint_e10.pth
```

## Results (10 epochs, see `log.txt`)

| Split | MAE (BPM) |
|-------|-----------|
| Train | 5.60 |
| Validation | 15.27 |
| Test | 8.17 |
