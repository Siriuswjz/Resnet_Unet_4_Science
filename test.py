# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import os
# import datetime
# from utils.config import *
# from src.model.ResNet_UNet import ResNet_UNet
# from src.data.PatchDataset import PatchDataset,Normalize
# from utils.extract_reconstruct_patches import reconstruct_from_patches,extract_patches_with_location
# from utils.losses import get_loss_function
# import numpy as np
# import matplotlib.pyplot as plt
# from utils.visualization_function.visualize_h5_data import visualize_prediction_data
# import h5py
#
# def main():
#     # 1. 设置设备
#     device = torch.device(DEVICE)
#     print(f"Using device: {device}")
#
#     # 2. 训练集和验证集 标准差和方差
#     print(f"Loading dataset from {DATA_DIR}...")
#     train_path = os.path.join(DATA_DIR, "yplus_1","train")
#     val_path = os.path.join(DATA_DIR, "yplus_1","val")
#     print(val_path)
#     normalize_feature = Normalize(DATA_MEAN_FEATURE, DATA_STD_FEATURE)
#     normalize_target = Normalize(DATA_MEAN_TARGET, DATA_STD_TARGET)
#
#     train_dataset = PatchDataset(train_path, transform_feature = normalize_feature, transform_target = normalize_target)
#     val_dataset =PatchDataset(val_path, transform_feature = normalize_feature, transform_target = normalize_target)
#
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=28,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )
#     print(f"Number of training samples: {len(val_loader)}")
#     for idx,(features,targets,locations) in enumerate(val_loader):
#         targets = targets.to(device)
#         targets_normalized = reconstruct_from_patches(pred_patches = targets,locations=locations,
#                                                full_shape=[INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH],device = device)
#         STD = DATA_STD_TARGET.to(device)
#         MEAN = DATA_MEAN_TARGET.to(device)
#         targets_raw = targets_normalized * STD + MEAN
#         if idx == 32:
#             output_dir = "./output/predictions"
#             targets_raw = targets_raw.to('cpu').numpy()
#             visualize_prediction_data(prediction_raw=targets_raw, idx=idx, output_dir=output_dir)
#             with h5py.File(os.path.join(output_dir, "predictions.h5"), "w") as hf:
#
#
#
# if __name__ == '__main__':
#     main()