import torch
import glob
import numpy as np
import os
from torch.utils.data import Dataset
from utils.extract_reconstruct_patches import extract_location_hw

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)


class PatchDataset(Dataset):
    def __init__(self, patch_dir,transform_feature=None,transform_target=None):
        self.files = sorted(glob.glob(os.path.join(patch_dir,"*" ,"*.npz")))
        self.transform_feature = transform_feature
        self.transform_target = transform_target

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        location = extract_location_hw(file_path)
        data = np.load(file_path)
        x = torch.from_numpy(data["feature"])  # [3, 256, 256]
        y = torch.from_numpy(data["target"])   # [3, 256, 256]
        if self.transform_feature:
            x = self.transform_feature(x)
        if self.transform_target:
            y = self.transform_target(y)
        return x.float(), y.float(),location