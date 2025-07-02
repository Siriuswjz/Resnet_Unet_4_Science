import re
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    logs_dir = "../output/logs"
    log_path = "training_log_20250630_222349.txt"
    log_path = os.path.join(logs_dir,log_path)
    losses_all = []
    losses_cf = []
    losses_qw = []
    losses_p = []


    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Train Loss CF:" in line:
            match = re.search(r"CF:\s*([\d\.]+),\s*QW:\s*([\d\.]+),\sP:\s*([\d\.]+)",line)
            if match:
                losses_cf.append(float(match.group(1)))
                losses_qw.append(float(match.group(2)))
                losses_p.append(float(match.group(3)))
        elif "Train Loss:" in line:
            match = re.search(r"Train Loss:\s*([\d\.]+)", line)
            if match:
                losses_all.append(float(match.group(1)))
        else:
            continue

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

    # 绘图
    print(f"检测结果losses_all: {len(losses_all)}")
    print(f"检测结果losses_cf: {len(losses_cf)}")
    print(f"检测结果losses_qw: {len(losses_qw)}")
    print(f"检测结果losses_p: {len(losses_p)}")
    epochs = list(range(1, len(losses_all) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses_all, label="Train Total Loss", color="blue", linewidth=2)
    plt.plot(epochs, losses_cf, label="Train CF Loss", linestyle="--", color="red")
    plt.plot(epochs, losses_qw, label="Train QW Loss", linestyle="--", color="green")
    plt.plot(epochs, losses_p, label="Train P Loss", linestyle="--", color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_detail.png")
    plt.show()
