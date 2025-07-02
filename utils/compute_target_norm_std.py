import numpy as np
import glob
import os
import h5py

def compute_mean_std(hdf5_paths,y_plus):
    Cf_vals, Qw_vals, P_vals = [], [], []

    print(f"Found {len(hdf5_paths)} hdf5 files...")

    for path in hdf5_paths:
        print(path)
        with h5py.File(path,'r') as f:
            group = f[y_plus]
            Cf_vals.append(group["friction_coefficient_2d"][:])
            Qw_vals.append(group["heat_flux_2d"][:])
            P_vals.append(group["pressure"][:])

    # 转为大数组
    Cf_all = np.concatenate([Cf.reshape(-1) for Cf in Cf_vals])
    Qw_all = np.concatenate([Qw.reshape(-1) for Qw in Qw_vals])
    P_all = np.concatenate([P.reshape(-1) for P in P_vals])

    # 计算均值和标准差
    Cf_mean, Cf_std = np.mean(Cf_all), np.std(Cf_all)
    Qw_mean, Qw_std = np.mean(Qw_all), np.std(Qw_all)
    P_mean, P_std = np.mean(P_all), np.std(P_all)

    print("\n--- Normalization Statistics ---")
    print(f"DATA_MEAN_TARGET = [{Cf_mean:.6f}, {Qw_mean:.6f}, {P_mean:.6f}]")
    print(f"DATA_STD_TARGET  = [{Cf_std:.6f}, {Qw_std:.6f}, {P_std:.6f}]")

if __name__ == "__main__":
    hdf5_root = "/data_8T/Jinzun/HDF5"
    hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root,"*.hdf5")))
    print(f"Found {len(hdf5_paths)} hdf5 files...") # 161

    y_plus_levels = ["yplus_wall_data", "yplus_1_data", "yplus_2_data", "yplus_5_data",
                     "yplus_10_data", "yplus_15_data", "yplus_30_data", "yplus_70_data",
                     "yplus_100_data", "yplus_200_data"]
    compute_mean_std(hdf5_paths,y_plus_levels[0])
