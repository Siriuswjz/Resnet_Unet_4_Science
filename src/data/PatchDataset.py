import torch
import glob
import numpy as np
import os
from torch.utils.data import Dataset

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)


class PatchDataset(Dataset):
    def __init__(self, patch_dir,transform=None):
        self.files = sorted(glob.glob(os.path.join(patch_dir, "*", "patch_*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.from_numpy(data["feature"])  # [3, 256, 256]
        y = torch.from_numpy(data["target"])   # [3, 256, 256]
        if self.transform:
            x = self.transform(x)
        return x.float(), y.float()