import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std+1e-8)

class HDF5Dataset(Dataset):
    """
    Args:
        y_plus_levels(list): 列表 ['yplus_wall_data'，'yplus_1_data',2,5,10,15,30,70,100,200]
    """
    def __init__(self, hdf5_dir, y_plus_levels,transform_feature=None,transform_target=None):
        self.transform_feature = transform_feature
        self.transform_target = transform_target
        self.y_plus_levels = y_plus_levels
        # 获取所有 HDF5 文件
        files = sorted(glob.glob(os.path.join(hdf5_dir, '*.hdf5')))
        num_files = len(files)
        split_idx = int(num_files * 0.8)
        self.files = files[split_idx:]

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.files)

    def __getitem__(self, idx):
        hdf5_path = self.files[idx]
        with h5py.File(hdf5_path, 'r') as f:
            group_name = self.y_plus_levels
            group = f[group_name]
            # 输入数据
            u = group['u'][:]
            v = group['v'][:]
            w = group['w'][:]
            # 将 u v w三个堆叠成 3 h w的维度
            features = np.stack([u, v, w], axis=0)

            # 目标输出数据
            wall_group_name = 'yplus_wall_data'
            wall_group = f[wall_group_name]
            target_cf = wall_group['friction_coefficient_2d'][:].astype(np.float32)
            target_qw = wall_group['heat_flux_2d'][:].astype(np.float32)
            target_p = wall_group['pressure'][:].astype(np.float32)
            targets = np.stack([target_cf, target_qw,target_p], axis=0)
            # 转成tensor
            x,y = torch.from_numpy(features).float(),torch.from_numpy(targets).float()
            if self.transform_feature is not None:
                x = self.transform_feature(x)
            if self.transform_target is not None:
                y = self.transform_target(y)
            return x, y

# if __name__ == "__main__":
#     # 假设 HDF5 文件存储在以下目录
#     hdf5_dir = "D:\AI Codes\Resnet_Unet\data\HDF5"
#     y_plus_levels = ['yplus_wall_data', 'yplus_1_data']
    # # 定义归一化参数（需要根据数据实际统计）
    # mean = torch.tensor([0.0, 0.0, 0.0])  # 对应 u, v, w 的均值
    # std = torch.tensor([1.0, 1.0, 1.0])  # 对应 u, v, w 的标准差
    # transform = Normalize(mean, std)

    # # 实例化 Dataset
    # dataset = HDF5Dataset(hdf5_dir,y_plus_levels=y_plus_levels)
    # print(dataset.__len__())
    #
    # # 使用 DataLoader
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # for features, targets in dataloader:
    #     print(f"Inputs shape:0 {features.shape}")
    #     print(f"Outputs shape: {targets.shape}")
    #     break