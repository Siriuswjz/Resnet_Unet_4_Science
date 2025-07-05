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
    pattern = '_(\d+)_(\d+)'
    dict_idx_truth_all = {}
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
            heat_flux_2d = group['heat_flux_2d'][:].astype(np.float32) # type: ignore
            p = group['pressure'][:].astype(np.float32) # type: ignore
            target = np.stack([friction_coefficient_2d, heat_flux_2d, p], axis=0)
            target_tensor = torch.from_numpy(target).float().to(device)
            dict_idx_truth_all[idx] = target_tensor
    return dict_idx_truth_all

def main():
    # 设备
    device = torch.device(DEVICE)

    # 模型
    model = ResNet_UNet(in_channels=INPUT_CHANNELS, num_classes=OUTPUT_CLASSES, backbone=BACKBONE_NAME).to(device)

    # 权重
    checkpoint_path = "./output/checkpoints/ResnetUnet_best_model_20250703_181153.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 输入预处理
    print(f"Using {INPUT_Y_TYPE} datas!!!!!!")
    feature_path = os.path.join(DATA_DIR, INPUT_Y_TYPE,"val")
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
    mae_fn = get_loss_function('mae', **loss_kwargs)

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
    dict_idx_truth_all = truth_fn(val_hdf5_paths, device)
    print("真实数据保存完毕")

    # 可视化输出目录
    output_dir = f"./output/error_visualization/{INPUT_Y_TYPE}_data"
    # 误差 mae分析目录
    output_dir_error =  f"./output/errors"
    total_cf,total_qw,total_p =0,0,0
    with open(os.path.join(output_dir_error, f"{INPUT_Y_TYPE}.txt"), "w") as f:
        for idx in dict_idx_predictions_all:
            prediction = dict_idx_predictions_all[idx]
            # if idx ==1490:
            #     visualize_prediction_data(prediction_raw=prediction.to('cpu').numpy(),idx=idx,output_dir=output_dir)
            truth = dict_idx_truth_all[idx]
            # if idx ==1490:
            #     visualize_prediction_data(prediction_raw=truth.to('cpu').numpy(),idx=idx,output_dir=output_dir)
            mae_cf = mae_fn(prediction[0], truth[0])
            mae_qw = mae_fn(prediction[1], truth[1])
            mae_p = mae_fn(prediction[2], truth[2])
            total_cf += mae_cf
            total_qw += mae_qw
            total_p += mae_p
            f.write(f'time:{idx}---mae_cf:{mae_cf:.6f}, mae_qw:{mae_qw:.6f}, mae_p:{mae_p:.6f}---\n')
            error_cf = torch.abs((truth[0]- prediction[0])) / mae_cf
            error_qw = torch.abs((truth[1]- prediction[1])) / mae_qw
            error_p = torch.abs((truth[2]- prediction[2])) / mae_p
            error_data = torch.stack([error_cf, error_qw, error_p], dim=0).to('cpu').numpy()
            visualize_error_data(error_data = error_data, idx = idx ,output_dir=output_dir,yplus = INPUT_Y_TYPE)
        total_cf/=len(predict_loader)
        total_qw/=len(predict_loader)
        total_p/=len(predict_loader)
        f.write(f'mean---mean_cf:{total_cf:.6f}, mean_qw:{total_qw:.6f}, mean_p:{total_p:.6f}\n')



if __name__ == "__main__":
    main()
