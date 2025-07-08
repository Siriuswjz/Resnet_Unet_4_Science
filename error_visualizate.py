import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils.config import *
from src.model.ResNet_UNet import ResNet_UNet
from src.data.PatchDataset import PatchDataset,Normalize
from utils.extract_reconstruct_patches import reconstruct_from_patches,extract_patches_with_location
from utils.visualization_function.visualize_h5_data import visualize_prediction_data, visualize_error_data,visualize_h5_data
import re
import glob
import numpy as np
from utils.losses import get_loss_function

pattern = '_(\d+)_(\d+)'

def predict_fn(loader, model, device):
    model.eval()
    indexes=[]
    for i in range(33):
        indexes.append(i)

    idx_start = 1426
    step = 2

    dict_idx_predictions_all = {}
    with torch.no_grad():
        for idx,(feature, _,location) in enumerate(loader):
            location = location.to(device)
            feature = feature.to(device)
            predictions = model(feature)
            predictions_normalized = reconstruct_from_patches(pred_patches=predictions, locations=location,
                                                              full_shape=[INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
                                                              device=device)
            STD = DATA_STD_TARGET.to(device)
            MEAN = DATA_MEAN_TARGET.to(device)
            predictions_raw = predictions_normalized * STD + MEAN  # [3,1400,800]
            idx_curr = idx_start + idx * step
            dict_idx_predictions_all[idx_curr] = predictions_raw
    return dict_idx_predictions_all

def truth_fn(hdf5_paths,device):
    dict_idx_truth_all = {}
    dict_idx_rms_all = {}
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, 'r') as f:
            # 得到索引 比如1490
            hdf5_path = os.path.basename(hdf5_path)
            match = re.search(pattern, hdf5_path)
            if match:
                idx = int(match.group(1))
                print(idx)
            else:
                print("匹配失败")

            # 组
            group = f["yplus_wall_data"]
            friction_coefficient_2d = group['friction_coefficient_2d'][:].astype(np.float32) # type: ignore
            h, w = friction_coefficient_2d.shape
            heat_flux_2d = group['heat_flux_2d'][:].astype(np.float32) # type: ignore
            p = group['pressure'][:].astype(np.float32) # type: ignore
            target = np.stack([friction_coefficient_2d, heat_flux_2d, p], axis=0)
            target_rms = np.sqrt(np.sum(target ** 2, axis=(1, 2)) / (h * w))
            dict_idx_truth_all[idx] = target
            dict_idx_rms_all[idx] = target_rms
    return dict_idx_truth_all, dict_idx_rms_all

def main():
    # 设备
    device = torch.device(DEVICE)

    # 模型
    model = ResNet_UNet(in_channels=INPUT_CHANNELS, num_classes=OUTPUT_CLASSES, backbone=BACKBONE_NAME).to(device)

    # 权重
    checkpoint_path = "./output/checkpoints/ResnetUnet_best_model_20250704_182554.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 输入预处理
    print(f"Using {INPUT_Y_TYPE} datas!!!!!!")
    feature_path = os.path.join(DATA_DIR, f'{INPUT_Y_TYPE}_data',"val")
    print(f"Feature path: {feature_path}")
    normalize_feature = Normalize(DATA_MEAN_FEATURE, DATA_STD_FEATURE)
    predict_dataset = PatchDataset(feature_path, transform_feature=normalize_feature)
    predict_loader = DataLoader(predict_dataset,
                                batch_size=28,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)
    print(f"Predict samples={len(predict_dataset)}")
    print(f"Predict loader batches: {len(predict_loader)}")

    dict_idx_predictions_all = predict_fn(predict_loader, model, device)
    print("预测完毕")

    # 误差函数 mse
    print(f"Using loss function: mae")
    loss_kwargs = {}
    loss_fn = get_loss_function('mae', **loss_kwargs)

    # hdf5目录
    os_name = os.name
    if os_name == 'nt':
        hdf5_root = "../../data/HDF5"
        hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root, "*.hdf5")))
    else:
        hdf5_root = "/data_8T/Jinzun/HDF5"
        hdf5_paths = sorted(glob.glob(os.path.join(hdf5_root, "*.hdf5")))
    # 得到测试集
    split_idx = int(len(hdf5_paths) * 0.8)
    val_hdf5_paths = hdf5_paths[split_idx:]
    dict_idx_truth_all,dict_idx_rms_all = truth_fn(val_hdf5_paths, device)
    print("真实数据保存完毕")

    # 可视化输出目录
    output_dir = f"./output/error_visualization/{INPUT_Y_TYPE}_data"
    # 误差 mae分析目录
    output_dir_error =  f"./output/errors"
    # 初始化总 MAE 累加器
    total_mae = np.zeros(3)  # 使用 NumPy 数组进行更简洁的累加

    with open(os.path.join(output_dir_error, f"{INPUT_Y_TYPE}_rms.txt"), "w") as f:
        # 遍历索引，获取预测值和真实值
        for idx in dict_idx_predictions_all:
            prediction_np = dict_idx_predictions_all[idx].detach().cpu().numpy()
            truth_np = dict_idx_truth_all[idx]
            rms_np = dict_idx_rms_all[idx]
            # if idx == 1490:
            #     visualize_prediction_data(prediction_raw=prediction_np, input_y_type=INPUT_Y_TYPE,idx=idx, output_dir=output_dir)

            # 计算绝对误差 三维
            error_abs = np.abs(prediction_np - truth_np)
            _, h, w = error_abs.shape

            # error_abs.sum(axis=(1, 2)) 对 H 和 W 维度进行求和 (结果: (3,))
            error_mae = (error_abs.sum(axis=(1, 2)) / (h * w))/ rms_np

            # 累加总 MAE
            total_mae += error_mae

            # 提取单个 MAE 值用于日志记录
            mae_cf, mae_qw, mae_p = error_mae
            max_error = error_abs.max()

            # 记录单个 MAE 值
            f.write(f'time:{idx}---mae_cf:{mae_cf:.6f}, mae_qw:{mae_qw:.6f}, mae_p:{mae_p:.6f}---\n')

            # # 对每个通道的误差进行归一化以便可视化
            # normalized_errors = []
            # for i in range(error_abs.shape[0]):  # 遍历通道 (0, 1, 2)
            #     curr_error = error_abs[i]
            #     max_error = curr_error.max()
            #     if max_error > 1e-6:
            #         normalized_error = curr_error / max_error
            #     else:
            #         normalized_error = np.zeros_like(curr_error)
            #
            #     normalized_errors.append(normalized_error)
            #
            # # 将归一化后的误差重新堆叠成一个 (3, H, W) 数组用于可视化
            # error_data_for_viz = np.stack(normalized_errors, axis=0)
            # visualize_error_data(error_data=error_abs, idx=idx, output_dir=output_dir, yplus=INPUT_Y_TYPE)

        # 计算总体平均 MAE total_mae 是一个 (3,) 形状的求和数组，除以样本数量
        mean_overall_mae = total_mae / len(predict_loader)
        mean_cf, mean_qw, mean_p = mean_overall_mae
        f.write(f'mean---mean_cf:{mean_cf:.6f}, mean_qw:{mean_qw:.6f}, mean_p:{mean_p:.6f}\n')
        print('fine')



if __name__ == "__main__":
    main()
