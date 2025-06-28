import numpy as np
import glob
import os
import h5py

def compute_mean_std(hdf5_paths,y_plus):
    u_vals, v_vals, w_vals = [], [], []

    print(f"Found {len(hdf5_paths)} hdf5 files...")

    for path in hdf5_paths:
        print(path)
        with h5py.File(path,'r') as f:
            group = f[y_plus]
            u_vals.append(group["u"][:])
            v_vals.append(group["v"][:])
            w_vals.append(group["w"][:])

    # 转为大数组
    u_all = np.concatenate([u.reshape(-1) for u in u_vals])
    v_all = np.concatenate([v.reshape(-1) for v in v_vals])
    w_all = np.concatenate([w.reshape(-1) for w in w_vals])

    # 计算均值和标准差
    u_mean, u_std = np.mean(u_all), np.std(u_all)
    v_mean, v_std = np.mean(v_all), np.std(v_all)
    w_mean, w_std = np.mean(w_all), np.std(w_all)

    print("\n--- Normalization Statistics ---")
    print(f"DATA_MEAN = [{u_mean:.6f}, {v_mean:.6f}, {w_mean:.6f}]")
    print(f"DATA_STD  = [{u_std:.6f}, {v_std:.6f}, {w_std:.6f}]")

if __name__ == "__main__":
    hdf5_root = "../data/HDF5"
    hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root,"*.hdf5")))
    print(f"Found {len(hdf5_paths)} hdf5 files...") # 161

    y_plus_levels = ["yplus_wall_data", "yplus_1_data", "yplus_2_data", "yplus_5_data",
                     "yplus_10_data", "yplus_15_data", "yplus_30_data", "yplus_70_data",
                     "yplus_100_data", "yplus_200_data"]
    compute_mean_std(hdf5_paths,y_plus_levels[1])
