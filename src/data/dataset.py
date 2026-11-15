import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ScalogramDataset(Dataset):
    def __init__(self, subject_list, data_dir):
        self.data_dir = data_dir
        self.subject_list = subject_list
        self.file_map = []

        # Build an index map: (subject_id, window_index_within_that_subject)
        for sub in self.subject_list:
            y_path = os.path.join(data_dir, f"{sub}_y.npy")
            if not os.path.exists(y_path):
                print(f"Warning: {y_path} not found. Skipping subject {sub}.")
                continue
           
            y = np.load(y_path)
            for i in range(len(y)):
                self.file_map.append((sub, i))
            del y # Keep RAM clean

        print(f"Dataset initialized with subjects {self.subject_list}")
        print(f"Total windows: {len(self.file_map)}")

    def __len__(self):
        return len(self.file_map)

    def __getitem__(self, idx):
        sub_id, window_idx = self.file_map[idx]

        # mmap_mode='r' reads just ONE window without loading the whole file
        X = np.load(os.path.join(self.data_dir, f"{sub_id}_X.npy"), mmap_mode='r')
        y = np.load(os.path.join(self.data_dir, f"{sub_id}_y.npy"), mmap_mode='r')

        sample = torch.from_numpy(X[window_idx].copy()).float()
        label = torch.tensor(y[window_idx]).float()

        return sample, label