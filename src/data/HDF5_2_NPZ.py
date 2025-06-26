import torch
from torch.utils.data import DataLoader,Dataset
import h5py
import numpy as np
import os
from utils.extract_reconstruct_patches import extract_patches_with_location,reconstruct_from_patches

def save_patches_from_hdf5(hdf5_path, save_dir, y_plus_levels,patch_size=256, stride=192):
    """
    Args:
        hdf5_path:
        save_dir:
        y_plus_levels:[yplus_wall_data,yplus_1_data,2,5,10,15,30,70,100,200]
    """
    os.makedirs(save_dir, exist_ok=True)

    # 读取 hdf5 中的 cf、qw、p
    with h5py.File(hdf5_path, 'r') as f:
        # 输入的u v w
        group_name = y_plus_levels[1]
        group = f[group_name]
        u = group['u'][:].astype(np.float32)
        v = group['v'][:].astype(np.float32)
        w  = group['w'][:].astype(np.float32)
        features = np.stack([u,v,w],axis=0) # [3,1400,800]
        features_tensor = torch.from_numpy(features).float()

        # 输出的 cf qw p
        group = f[y_plus_levels[0]]
        friction_coefficient_2d = group['friction_coefficient_2d'][:].astype(np.float32)
        heat_flux_2d = group['heat_flux_2d'][:].astype(np.float32)
        p = group['pressure'][:].astype(np.float32)
        target = np.stack([friction_coefficient_2d,heat_flux_2d,p],axis=0)
        target_tensor = torch.from_numpy(target).float()

        # patches
        _,locations = extract_patches_with_location(features_tensor)

        count = 0
        for (h,w) in locations:
            feature_patch = features_tensor[:,h:h+patch_size,w:w+patch_size]
            target_patch = target_tensor[:,h:h+patch_size,w:w+patch_size]
            filename = f"patch_{h}_{w}.npz"
            path = os.path.join(save_dir, filename)
            np.savez(path,feature=feature_patch,target=target_patch)
            count += 1
        print(f"saved {count} patch pairs")

if __name__ == "__main__":
    hdf5_dir = "../../data/HDF5/compressible_channel_flow_data_1490_1492.hdf5"
    save_dir = "../../data/NPZ/1490_1492"
    y_plus_levels = ["yplus_wall_data","yplus_1_data","yplus_2_data","yplus_5_data","yplus_10_data","yplus_15_data",
                     "yplus_30_data","yplus_70_data","yplus_100_data","yplus_200_data"]
    save_patches_from_hdf5(hdf5_dir,save_dir,y_plus_levels)



