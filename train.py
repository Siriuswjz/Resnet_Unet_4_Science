# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import datetime
from utils.config import *
from src.model.ResNet_UNet import ResNet_UNet
from src.data.PatchDataset import PatchDataset
from utils.extract_reconstruct_patches import reconstruct_from_patches,extract_patches_with_location
from utils.losses import get_loss_function
import numpy as np

# --- 辅助函数：保存模型检查点 ---
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    保存模型和优化器的状态。
    state (dict): 包含模型、优化器、epoch等信息的字典。
    filename (str): 保存的文件名。
    """
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    print(f"Saving checkpoint to {filepath}")
    torch.save(state, filepath)


# --- 训练函数 ---
def train_fn(loader, model, optimizer, loss_fn, scaler, device, scheduler=None):
    """
    执行一个 epoch 的训练。
    loader (DataLoader): 训练数据加载器。
    model (nn.Module): 待训练的模型。
    optimizer (Optimizer): 优化器。
    loss_fn (nn.Module): 损失函数。
    scaler (torch.cuda.amp.GradScaler): 混合精度训练的 GradScaler。
    device (torch.device): 训练设备 (CPU 或 CUDA)。
    scheduler (LRScheduler, optional): 学习率调度器。
    """
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0

    model.train()  # 设置模型为训练模式

    for batch_idx, (features, targets) in enumerate(loop):
        features = features.to(device)
        targets = targets.to(device)

        # 混合精度训练
        with torch.cuda.amp.autocast():
            predictions = model(features)
            loss = loss_fn(predictions, targets)

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # 更新进度条
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch Avg Training Loss: {avg_loss:.4f}")

    if scheduler is not None:
        scheduler.step()  # 学习率调度器更新
        print(f"Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

    return avg_loss

def eval_fn(loader, model, loss_fn, device,locations):
    """
    执行一个 epoch 的验证。
    loader (DataLoader): 验证数据加载器。
    model (nn.Module): 待评估的模型。
    loss_fn (nn.Module): 损失函数。
    device (torch.device): 评估设备 (CPU 或 CUDA)。
    """
    loop = tqdm(loader, desc="Validation")
    total_loss = 0.0

    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 在评估模式下禁用梯度计算
        for features, targets in loop:
            assert features.shape[0] == len(locations), "Batch size 必须等于每张图的 patch 数"
            features = features.to(device)
            predictions = model(features)

            targets = targets.to(device)
            targets = reconstruct_from_patches(pred_patches = targets,locations=locations,
                                               full_shape=[INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH])
            predictions = reconstruct_from_patches(pred_patches = predictions, locations=locations,
                                                full_shape=[INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH])

            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch Avg Validation Loss: {avg_loss:.4f}")

    return avg_loss


# --- 主函数 ---
def main():
    # 1. 设置设备
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 2. 训练集和验证集
    print(f"Loading dataset from {DATA_DIR}...")
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")
    train_dataset = PatchDataset(train_path)
    val_dataset =PatchDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True  # 允许 PyTorch 在 GPU 上更快地传输数据
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=28,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Dataset split: Train samples={len(train_dataset)}, Validation samples={len(val_dataset)}")
    print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    # 3. 模型实例化
    print(f"Initializing model: {MODEL_NAME} with backbone {BACKBONE_NAME}...")
    model = ResNet_UNet(in_channels=INPUT_CHANNELS, num_classes=OUTPUT_CLASSES, backbone=BACKBONE_NAME,
                 pretrained=PRETRAINED_ENCODER)
    model.to(device)
    print(f"Model on {DEVICE}")

    # 4. 损失函数
    print(f"Using loss function: {LOSS_FN_TYPE}")
    loss_kwargs = {}
    loss_fn = get_loss_function(LOSS_FN_TYPE, **loss_kwargs)

    # 5. 优化器
    print(f"Using optimizer: {OPTIMIZER_TYPE} with learning rate {LEARNING_RATE}")
    if OPTIMIZER_TYPE.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_TYPE.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_TYPE}")

    # 6. 学习率调度器
    # scheduler = None
    # if USE_SCHEDULER:
    #     scheduler_config = SCHEDULER_CONFIG.get(SCHEDULER_TYPE)
    #     if scheduler_config is None:
    #         raise ValueError(f"Scheduler config for {SCHEDULER_TYPE} not found in config.py")
    #
    #     print(f"Using scheduler: {SCHEDULER_TYPE}")
    #     if SCHEDULER_TYPE == "CosineAnnealingLR":
    #         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                          T_max=scheduler_config["T_max"],
    #                                                          eta_min=scheduler_config["eta_min"])
    #     elif SCHEDULER_TYPE == "StepLR":
    #         scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                               step_size=scheduler_config["step_size"],
    #                                               gamma=scheduler_config["gamma"])
    #     else:
    #         print(
    #             f"Warning: Scheduler type {SCHEDULER_TYPE} is not explicitly handled, using default PyTorch behavior.")
    #         # For other schedulers, you might need to add specific instantiation logic

    # 7. 混合精度训练 Scaler
    scaler = torch.cuda.amp.GradScaler()

    # 8. 训练循环
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---")
    best_val_loss = float('inf')
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_DIR, f"training_log_{start_time}.txt")

    # 8. locations
    img = torch.randn(1, 1400, 800)  # [C, H, W]，比如 1 通道预测热图
    _, locations = extract_patches_with_location(img, patch_size=256, stride=192)

    with open(log_file_path, 'w') as log_f:
        log_f.write(f"Training started at: {start_time}\n")
        log_f.write(f"Configuration: {globals()}\n")
        log_f.write("-" * 50 + "\n")

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            log_f.write(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}\n")

            # 训练阶段
            train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
            log_f.write(f"Train Loss: {train_loss:.4f}\n")

            # 验证阶段
            if (epoch + 1) % EVAL_INTERVAL == 0:
                val_loss = eval_fn(val_loader, model, loss_fn, device,locations)
                log_f.write(f"Validation Loss: {val_loss:.4f}\n")

                # 保存最佳模型
                if SAVE_BEST_MODEL and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model...")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, filename=f"{MODEL_NAME}_best_model_{start_time}.pth.tar")
            else:
                print(f"Skipping validation for this epoch (eval interval is {EVAL_INTERVAL}).")

            # 每隔一段时间保存一个通用检查点（可选）
            # if (epoch + 1) % 5 == 0:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }, filename=f"{MODEL_NAME}_epoch_{epoch+1}.pth.tar")

        print("\n--- Training Finished ---")
        log_f.write("\n--- Training Finished ---\n")


if __name__ == "__main__":
    # 设置随机种子，确保可复现性
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 如果输入尺寸固定，可以设为True以加速

    main()