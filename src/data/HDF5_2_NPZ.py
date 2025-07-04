import torch
from torch.utils.data import DataLoader,Dataset
import h5py
import numpy as np
import os
from utils.extract_reconstruct_patches import extract_patches_with_location,reconstruct_from_patches
import glob

os_name = os.name

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
        # 输入的u v w yplus_10_data
        group_name = y_plus_levels[5]
        print(f"Using {group_name}")
        group = f[group_name]
        u = group['u'][:].astype(np.float32)
        v = group['v'][:].astype(np.float32)
        w  = group['w'][:].astype(np.float32)
        features = np.stack([u,v,w],axis=0) # [3,1400,800]
        features_tensor = torch.from_numpy(features).float()

        # 输出的 cf qw p yplus_wall_data
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
    y_plus_levels = ["yplus_wall_data", "yplus_1_data", "yplus_2_data", "yplus_5_data",
                     "yplus_10_data", "yplus_15_data", "yplus_30_data", "yplus_70_data",
                     "yplus_100_data", "yplus_200_data"]

    yplus = y_plus_levels[5]
    if os_name == 'nt':
        hdf5_root = "../../data/HDF5"
        save_root = f"../../data/NPZ/{yplus}"
        # 搜索所有 HDF5 文件
        hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root, "*.hdf5")))


    else:
        hdf5_root = "/data_8T/Jinzun/HDF5"
        save_root = f"/data_8T/Jinzun/NPZ/{yplus}"
        # 搜索所有 HDF5 文件
        hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root, "*.hdf5")))

    # 划分 80% train + 20% val
    num_files = len(hdf5_paths)
    split_idx = int(num_files * 0.8)

    train_paths = hdf5_paths[:split_idx]
    val_paths = hdf5_paths[split_idx:]

    print(f"Total files: {num_files}")
    print(f"Train files: {len(train_paths)}")
    print(f"Val files: {len(val_paths)}")

    # 创建子目录
    train_dir = os.path.join(save_root, "train")
    val_dir = os.path.join(save_root, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 处理 train
    for hdf5_path in train_paths:
        file_name = os.path.basename(hdf5_path).replace("compressible_channel_flow_data_", "").replace(".hdf5", "")
        save_dir = os.path.join(train_dir, file_name)

        print(f"[Train] Processing {hdf5_path} -> {save_dir}")
        save_patches_from_hdf5(hdf5_path, save_dir, y_plus_levels)

    # 处理 val
    for hdf5_path in val_paths:
        file_name = os.path.basename(hdf5_path).replace("compressible_channel_flow_data_", "").replace(".hdf5", "")
        save_dir = os.path.join(val_dir, file_name)

        print(f"[Val] Processing {hdf5_path} -> {save_dir}")
        save_patches_from_hdf5(hdf5_path, save_dir,y_plus_levels)





