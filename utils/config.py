import os
import torch

# --- 项目根目录设置 ---
# 获取当前文件所在目录的父目录，作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # utils
BASE_DIR = os.path.dirname(BASE_DIR) # Resnet_Unet

# --- 通用设置 ---
PROJECT_NAME = "Resnet_Unet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42 # 随机种子，用于复现性
NUM_WORKERS = 4

# --- 数据集设置 ---
DATA_DIR = os.path.join(BASE_DIR, "data","NPZ") # 数据存放目录

INPUT_HEIGHT = 1400
INPUT_WIDTH = 800
INPUT_CHANNELS = 3
OUTPUT_CLASSES = 3

# 数据归一化参数 (根据你的数据集的实际统计值进行调整)
DATA_MEAN = [0.5, 0.5, 0.5, 0.5] # 每个输入通道的均值
DATA_STD = [0.5, 0.5, 0.5, 0.5]  # 每个输入通道的标准差

# --- 模型设置 ---
MODEL_NAME = "ResnetUnet"
BACKBONE_NAME = "resnet50" # 编码器骨干网络
PRETRAINED_ENCODER = False

# --- 训练设置 ---
# BATCH_SIZE = 32 # 训练批次大小
LEARNING_RATE = 1e-4 # 初始学习率
NUM_EPOCHS = 100 # 训练轮数
OPTIMIZER_TYPE = "Adam"
LOSS_FN_TYPE = "mse"
# HUBER_LOSS_BETA = 1.0 # 如果使用Huber Loss，设置beta值

# 学习率调度器设置
# USE_SCHEDULER = True
# SCHEDULER_TYPE = "CosineAnnealingLR" # 例如 CosineAnnealingLR, StepLR, ReduceLROnPlateau
# SCHEDULER_CONFIG = {
#     "CosineAnnealingLR": {"T_max": NUM_EPOCHS, "eta_min": 1e-6},
#     "StepLR": {"step_size": 20, "gamma": 0.1},
# }

# --- 验证/评估设置 ---
SAVE_BEST_MODEL = True # 是否保存性能最好的模型
EVAL_INTERVAL = 5 # 每N个epoch进行一次验证

# --- 日志和保存路径 ---
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# 确保输出目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print(f"Loaded configuration for {PROJECT_NAME}")
print(f"Running on device: {DEVICE}")