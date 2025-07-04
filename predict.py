import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utils.config import *
from src.model.ResNet_UNet import ResNet_UNet
from src.data.PatchDataset import PatchDataset,Normalize
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

    # 确保npz文件夹存在
    output_npz_dir = './output/predictions/npz'
    os.makedirs(output_npz_dir, exist_ok=True)

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
            np.savez(f'{output_npz_dir}/{idx_curr}.npz',prediction=predictions_raw.cpu().numpy())
            print(f"saved {idx_curr}.npz")
    return dict_idx_predictions_all

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
    # 可视化输出目录
    output_dir = f"./output/predictions/{INPUT_Y_TYPE}_data"
    for idx,prediction in dict_idx_predictions_all.items():
        prediction = prediction.to('cpu').numpy()
        if idx == 1448:
            visualize_prediction_data(prediction_raw = prediction,idx = idx ,output_dir=output_dir)
        else:
            visualize_prediction_data(prediction_raw = prediction,idx = idx,output_dir=output_dir)

if __name__ == "__main__":
    main()





