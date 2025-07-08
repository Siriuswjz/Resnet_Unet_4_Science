import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils.config import *
from src.model.ResNet_UNet import ResNet_UNet
from src.data.HDF5Dataset import HDF5Dataset ,Normalize
from utils.extract_reconstruct_patches import reconstruct_from_patches,extract_patches_with_location
from utils.visualization_function.visualize_h5_data import visualize_prediction_data
import re
import numpy as np

def predict_fn(loader, model, device):
    model.eval()
    indexes=[]
    for i in range(33):
        indexes.append(i)

    idx_start = 1426
    step = 2

    dict_idx_predictions_all = {}
    dict_idx_truth_all = {}
    # 确保npz文件夹存在
    # output_npz_dir = './output/predictions/npz'
    # os.makedirs(output_npz_dir, exist_ok=True)

    with torch.no_grad():
        for idx,(feature, target) in enumerate(loader):
            feature = feature.to(device)
            predictions = model(feature)
            STD = DATA_STD_TARGET.to(device)
            MEAN = DATA_MEAN_TARGET.to(device)
            predictions_raw = predictions * STD + MEAN  # [3,1400,800]
            idx_curr = idx_start + idx * step
            dict_idx_predictions_all[idx_curr] = predictions_raw.detach().cpu().numpy().squeeze(0)
            dict_idx_truth_all[idx_curr] = target.detach().cpu().numpy().squeeze(0)
            # np.savez(f'{output_npz_dir}/{idx_curr}.npz',prediction=predictions_raw.cpu().numpy())
            # print(f"saved {idx_curr}.npz")
    return dict_idx_predictions_all,dict_idx_truth_all

def error_fn(prediction, target):
    _,h,w = prediction.shape
    target_rms = np.sqrt(np.sum(target ** 2, axis=(1, 2)) / (h * w))
    error_abs = np.abs(prediction - target)
    error_rms = (error_abs.sum(axis=(1, 2)) / (h * w))/ target_rms
    return error_rms


def main():
    device = torch.device(DEVICE)
    model = ResNet_UNet(in_channels=INPUT_CHANNELS, num_classes=OUTPUT_CLASSES, backbone='resnet50').to(device)
    # 权重
    checkpoint_path = "./output/checkpoints/ResnetUnet_yplus100_20250704_182554.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 输入预处理
    y_plus_levels = ["yplus_wall_data", "yplus_1_data", "yplus_2_data", "yplus_5_data",
                     "yplus_10_data", "yplus_15_data", "yplus_30_data", "yplus_70_data",
                     "yplus_100_data", "yplus_200_data"]
    feature_path = '/data_8T/Jinzun/HDF5'
    print(f"Feature path: {feature_path}")
    print(f'Using {INPUT_Y_TYPE} data')
    normalize_feature = Normalize(DATA_MEAN_FEATURE, DATA_STD_FEATURE)
    predict_dataset = HDF5Dataset(hdf5_dir=feature_path,y_plus_levels=f'{INPUT_Y_TYPE}_data', transform_feature=normalize_feature)
    predict_loader = DataLoader(predict_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)
    print(f"Predict samples={len(predict_dataset)}")
    print(f"Predict loader batches: {len(predict_loader)}")

    dict_idx_predictions_all,dict_idx_truth_all = predict_fn(predict_loader, model, device)
    print("预测完毕")
    extrema_dict = {'yplus_30_1490': [[-0.001,0.022],[-0.000,0.011],[0.152,0.254]],
                    'yplus_30_1448':[[-0.002,0.025],[-0.001,0.012],[0.142,0.240]],
                    'yplus_15_1490': [[-0.001,0.022],[-0.000,0.011],[0.152,0.254]],
                    'yplus_15_1448':[[-0.002,0.025],[-0.001,0.012],[0.142,0.240]],
                    'yplus_100_1490':[[-0.001,0.022],[-0.000,0.011],[0.152,0.254]],
                    'yplus_100_1448':[[-0.002,0.025],[-0.001,0.012],[0.142,0.240]]}
    # 预测输出目录
    output_dir = f"./output/predictions/yplus100_1400_800_data"
    # 误差输出目录
    output_dir_error = f"./output/errors_no_patch"
    os.makedirs(output_dir_error,exist_ok=True)
    total_rms = np.zeros(3)
    with open(os.path.join(output_dir_error, f"{INPUT_Y_TYPE}_rms.txt"), 'w') as f:
        for idx in dict_idx_predictions_all.keys():
            prediction = dict_idx_predictions_all[idx]
            truth = dict_idx_truth_all[idx]
            error_rms = error_fn(prediction, truth)
            total_rms += error_rms
            rms_cf, rms_qw, rms_p = error_rms
            f.write(f'time: {idx}---rms_cf:{rms_cf:.6f}, rms_qw:{rms_qw:.6f}, rms_p:{rms_p:.6f}---\n')
            if idx==1490:
                visualize_prediction_data(prediction_raw = prediction,idx=idx,input_y_type=INPUT_Y_TYPE,
                output_dir=output_dir,extrema=extrema_dict['yplus_30_1490'])
        mean_overall_rms = total_rms / len(predict_dataset)
        mean_cf, mean_qw, mean_p = mean_overall_rms
        f.write(f'mean---mean_cf:{mean_cf:.6f}, mean_qw:{mean_qw:.6f}, mean_p:{mean_p:.6f}\n')


if __name__ == "__main__":
    main()
