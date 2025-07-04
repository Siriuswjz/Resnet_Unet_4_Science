# ResNet-UNet for Y+ Layer Feature Prediction

本项目基于 ResNet-UNet 架构，实现了对流体力学中 y+ 层特征的预测。项目包含数据预处理、特征归一化、模型训练与预测等完整流程。

## 目录结构

```
Resnet_Unet/
├── data/                        # 原始数据存放目录
├── output/                      # 输出结果目录
├── src/
│   ├── data/                    # 数据处理相关脚本
│   └── model/                   # 模型结构相关脚本
├── utils/                       # 工具函数与可视化
├── train.py                     # 训练主脚本
├── predict.py                   # 预测主脚本
├── test.py                      # 测试脚本
├── README.md                    # 项目说明文件
```

## 环境依赖

- Python 3.7+
- 推荐使用虚拟环境（如 venv 或 conda）

常用依赖包（请根据实际代码补充）：

```bash
pip install numpy torch h5py matplotlib
```

## 数据准备流程

1. **准备 HDF5 数据**  
   将原始数据放入 `data/` 目录。

2. **HDF5 转 NPZ**  
   使用 `src/data/HDF5_2_NPZ.py` 脚本，将 HDF5 数据转换为 yplus 层特征和 yplus wall 目标。注意需要修正 layer 23、27 和 58。

3. **特征归一化**  
   使用 `utils/compute_feature_norm_std.py` 计算特征的均值和标准差，并将结果复制到 `utils/config.py` 配置文件中。

4. **修正输入类型**  
   根据需要，修正 layer 20 的 `INPUT_Y_TYPE`。

## 训练流程

运行以下命令开始训练：

```bash
python train.py
```

训练过程中会自动保存模型和损失曲线等结果到 `output/` 目录。

## 预测流程

训练完成后，可使用以下命令进行预测：

```bash
python predict.py
```

预测结果将保存在 `output/` 目录。

## 主要脚本说明

- `train.py`：模型训练主脚本。
- `predict.py`：模型预测主脚本。
- `src/data/HDF5_2_NPZ.py`：HDF5 数据转换为 NPZ 格式。
- `utils/compute_feature_norm_std.py`：计算特征归一化参数。
- `utils/config.py`：配置文件，存储归一化参数等。
- `src/model/ResNet_UNet.py`：ResNet-UNet 主体结构。
- `utils/visualization.py` 及 `utils/visualization_function/`：可视化相关脚本。

## 可视化与分析

- `utils/draw_losses_carve.py`：绘制损失曲线。
- `error_visualizate.py`：误差可视化脚本。

## 备注

- 请根据实际数据和需求，适当修改脚本中的参数和路径。
- 如有问题请联系项目维护者。
