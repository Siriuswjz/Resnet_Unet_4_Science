import numpy as np
import glob
import os

def compute_mean_std(patch_dir):
    u_vals, v_vals, w_vals = [], [], []

    patch_paths = sorted(glob.glob(os.path.join(patch_dir, "**", "*.npz"), recursive=True))
    print(f"Found {len(patch_paths)} patches...")

    for path in patch_paths:
        data = np.load(path)
        feature = data['feature']  # shape: [3, H, W]
        u_vals.append(feature[0])  # u
        v_vals.append(feature[1])  # v
        w_vals.append(feature[2])  # w

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
    patch_dir = "../data/NPZ/yplus_1/train"
    compute_mean_std(patch_dir)
