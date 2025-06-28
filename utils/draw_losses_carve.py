import re
import matplotlib.pyplot as plt
import os

def draw_losses_carve_from_list(train_losses, val_losses, epoch):
    return

if __name__ == "__main__":
    logs_dir = "../output/logs"
    log_path = "training_log_20250627_184511.txt"
    log_path = os.path.join(logs_dir,log_path)
    train_losses = []
    # val_losses = []
    # val_epochs = []

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Train Loss:" in line:
            match = re.search(r"Train Loss:\s*([\d\.]+)", line)
            if match:
                train_losses.append(float(match.group(1)))
        # 提取Validation Loss
        # if "Validation Loss:" in line:
        #     match = re.search(r"Validation Loss:\s*([\d\.]+)", line)
        #     if match:
        #         val_losses.append(float(match.group(1)))
        #         # 同时记录它发生在第几个epoch
        #         # 往前找Epoch行
        #         for j in range(i, 0, -1):
        #             if "Epoch" in lines[j]:
        #                 epoch_match = re.search(r"Epoch\s+(\d+)/", lines[j])
        #                 if epoch_match:
        #                     val_epochs.append(int(epoch_match.group(1)))
        #                 break

    # 检查结果
    print("Train Losses:", train_losses)
    # print("Validation Losses:", val_losses)
    # print("Validation Epochs:", val_epochs)

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker="o", label="Train Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
