import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils.config import *
from src.model.ResNet_UNet import ResNet_UNet
from src.data.PatchDataset import PatchDataset,Normalize
from utils.extract_reconstruct_patches import reconstruct_from_patches,extract_patches_with_location
from utils.visualization_function.visualize_h5_data import visualize_prediction_data, visualize_error_data
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
            idx_curr = idx_start + idx*step
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
            friction_coefficient_2d = group['friction_coefficient_2d'][:].astype(np.float32)
            heat_flux_2d = group['heat_flux_2d'][:].astype(np.float32)
            p = group['pressure'][:].astype(np.float32)
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

    # 损失函数 mse
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
    dict_idx_truth_all = truth_fn(hdf5_paths, device)
    print("真实数据保存完毕")

    # 可视化输出目录
    output_dir = f"./output/error_visualization/{INPUT_Y_TYPE}_data"
    for idx in dict_idx_predictions_all:
        if idx == 1448:
            prediction = dict_idx_predictions_all[idx]
            truth = dict_idx_truth_all[idx]
            error_cf = (truth[0]- prediction[0]) / loss_fn(truth[0], prediction[0])
            error_qw = (truth[1]- prediction[1]) / loss_fn(truth[1], prediction[1])
            error_p = (truth[2]- prediction[2]) / loss_fn(truth[2], prediction[2])
            error_data = torch.stack([error_cf, error_qw, error_p], dim=0).to('cpu').numpy()
            visualize_error_data(error_data = error_data, idx = idx ,output_dir=output_dir,yplus = INPUT_Y_TYPE)



if __name__ == "__main__":
    main()
