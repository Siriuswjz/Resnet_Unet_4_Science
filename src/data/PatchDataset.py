import torch
import glob
import numpy as np
import os
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, patch_dir):
        self.files = sorted(glob.glob(os.path.join(patch_dir, "*", "patch_*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.from_numpy(data["feature"])  # [3, 256, 256]
        y = torch.from_numpy(data["target"])   # [3, 256, 256]
        return x.float(), y.float()