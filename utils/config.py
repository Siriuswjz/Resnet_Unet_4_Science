import os
import torch

# 项目根目录设置
os_name = os.name
if os_name == 'posix':
    DATA_DIR = "/data_8T/Jinzun/NPZ"
    BASE_DIR = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science"
else:
    DATA_DIR = "D:\AI Codes\Resnet_Unet\data\\NPZ"
    BASE_DIR = "D:\AI Codes\Resnet_Unet"
print(os_name)
# --- 通用设置 ---
PROJECT_NAME = "Resnet_Unet"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
SEED = 42 # 随机种子，用于复现性
NUM_WORKERS = 4

# 输入属性
INPUT_Y_TYPE = "yplus_15"
INPUT_HEIGHT = 1400
INPUT_WIDTH = 800
INPUT_CHANNELS = 3
OUTPUT_CLASSES = 3

# 归一化参数feature
dict_feature = {'yplus_1':[[0.045352, -0.000001, -0.000010],
                           [0.018032, 0.000491, 0.006800]],
                'yplus_5':[[0.155256, -0.000038, -0.000034],
                           [0.056940, 0.001948, 0.019795]],
                'yplus_15':[[0.378425, -0.000372, -0.000095],
                            [0.118740, 0.008565, 0.038349]],
                'yplus_30': [[0.577761, -0.000800, -0.000184],
                             [0.149489, 0.019171, 0.052471]],
                'yplus_70':[[0.784640, -0.000958, -0.000196],
                            [0.139977, 0.039637, 0.066212]],
                'yplus_100':[[0.842697, -0.000849, -0.000099],
                             [0.124568, 0.047473, 0.068853]]}

DATA_MEAN_FEATURE = torch.tensor(dict_feature[INPUT_Y_TYPE][0]).view(3,1,1)
DATA_STD_FEATURE =  torch.tensor(dict_feature[INPUT_Y_TYPE][1]).view(3,1,1)

# 归一化参数target
DATA_MEAN_TARGET = [0.006131, 0.003090, 0.198331]
DATA_STD_TARGET  = [0.002558, 0.001295, 0.006957]
DATA_MEAN_TARGET = torch.tensor(DATA_MEAN_TARGET).view(3,1,1)
DATA_STD_TARGET = torch.tensor(DATA_STD_TARGET).view(3,1,1)

# --- 模型设置 ---
MODEL_NAME = "ResnetUnet"
BACKBONE_NAME = "resnet50" # 编码器骨干网络
PRETRAINED_ENCODER = False

# --- 训练设置 ---
# BATCH_SIZE = 32 # 训练批次大小
LEARNING_RATE = 1e-3 # 初始学习率
NUM_EPOCHS = 200 # 训练轮数
OPTIMIZER_TYPE = "Adam"
LOSS_FN_TYPE = "mse"

# 学习率调度器设置
USE_SCHEDULER = True
SCHEDULER_TYPE = "StepLR" # 例如 CosineAnnealingLR, StepLR, ReduceLROnPlateau
SCHEDULER_CONFIG = {
    "CosineAnnealingLR": {"T_max": NUM_EPOCHS, "eta_min": 1e-6},
    "StepLR": {"step_size": 20, "gamma": 0.1},
}

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
