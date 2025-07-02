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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def predict_fn(loader, model, device, locations):
    model.eval()

    predictions_all = []
    with torch.no_grad():
        for feature, _ in loader:
            feature = feature.to(device)
            predictions = model(feature)
            predictions_normalized = reconstruct_from_patches(pred_patches=predictions, locations=locations,
                                                              full_shape=[INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
                                                              device=device)
            STD = DATA_STD_TARGET.to(device)
            MEAN = DATA_MEAN_TARGET.to(device)
            predictions_raw = predictions_normalized * STD + MEAN  # [3,1400,800]
            predictions_all.append(predictions_raw)
    return predictions_all

def main():
    # 设备
    device = torch.device(DEVICE)

    # 模型
    model = ResNet_UNet(in_channels=INPUT_CHANNELS, num_classes=OUTPUT_CLASSES, backbone=BACKBONE_NAME).to(device)

    # 权重
    checkpoint_path = "./output/checkpoints/ResnetUnet_best_model_20250630_222349.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 输入预处理
    feature_path = os.path.join(DATA_DIR, "yplus_1/val/1490_1492")
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

    # 文件名 用来绘图命名
    pattern = "(\d+-\d+)$"
    idx = re.search(pattern, feature_path)

    # locations
    img = torch.randn(1, 1400, 800)
    _, locations = extract_patches_with_location(img)

    predictions_all = predict_fn(predict_loader, model, device, locations)
    print("预测完毕")
    # 可视化输出目录
    output_dir = "D:/AI Codes/Resnet_Unet/output/predictions"
    for prediction in predictions_all:
        prediction = prediction.to('cpu').numpy()
        visualize_prediction_data(prediction_raw = prediction,idx = idx ,output_dir=output_dir)

if __name__ == "__main__":
    main()





