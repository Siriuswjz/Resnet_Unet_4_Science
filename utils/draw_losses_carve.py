import re
import matplotlib.pyplot as plt
import os

def plot_train(log_path,input_yplus):
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
    plt.title(f"yplus = {input_yplus} training loss per epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_name = f"yplus_{input_yplus}_training_loss.png"
    png_path = os.path.join("../output/plots/",png_name)
    plt.savefig(png_path)
    plt.show()

def plot_val(log_path,input_yplus):
    losses_all = []
    losses_cf = []
    losses_qw = []
    losses_p = []
    with open(log_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Validation Loss CF:" in line:
            match = re.search("CF:\s([\d\.]+),\s*QW:\s*([\d\.]+),\sP:\s([\d\.]+)",line)
            if match:
                losses_cf.append(float(match.group(1)))
                losses_qw.append(float(match.group(2)))
                losses_p.append(float(match.group(3)))
        elif "Validation Loss:" in line:
            match = re.search("Validation Loss:\s*([\d\.]+)", line)
            if match:
                losses_all.append(float(match.group(1)))
        else:
            continue
    print(f"检测结果losses_all: {len(losses_all)}")
    print(f"检测结果losses_cf: {len(losses_cf)}")
    print(f"检测结果losses_qw: {len(losses_qw)}")
    print(f"检测结果losses_p: {len(losses_p)}")
    epochs = list(range(1, len(losses_all) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses_all, label="Validation Total Loss", color="blue", linewidth=2)
    plt.plot(epochs, losses_cf, label="Validation CF Loss", linestyle="--", color="red")
    plt.plot(epochs, losses_qw, label="Validation QW Loss", linestyle="--", color="green")
    plt.plot(epochs, losses_p, label="Validation P Loss", linestyle="--", color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"yplus = {input_yplus} Validation loss per 5 epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_name = f"yplus_{input_yplus}_validation_loss.png"
    png_path = os.path.join("../output/plots/",png_name)
    plt.savefig(png_path)
    plt.show()





if __name__ == "__main__":
    logs_dir = "../output/logs"
    log_path = "training_log_20250709_203827.txt"
    input_yplus = 15
    log_path = os.path.join(logs_dir,log_path)
    # plot_train(log_path,input_yplus)
    plot_val(log_path,input_yplus)

