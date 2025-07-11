import torch
import os
from utils.config import *
import re
import numpy as np
import h5py
from matplotlib import pyplot as plt
from utils.visualization_function.plot_2d_counter import plot_2d_counter
from utils.visualization_function.read_grid import read_grid

os_name =os.name
if os_name == "nt":
    grid_x = "D:\AI Codes\Resnet_Unet\dxg.dat"
    grid_y = "D:\AI Codes\Resnet_Unet\dyg.dat"
    grid_z = "D:\AI Codes\Resnet_Unet\dzg.dat"
else:
    grid_x = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science/dxg.dat"
    grid_y = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science/dyg.dat"
    grid_z = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science/dzg.dat"
# 网格
mesh = read_grid(grid_x, grid_y, grid_z)

# 真实数据可视化绘图
def visualize_input_data(h5_path,yplus, idx, output_dir=None):
    """
    Loads data from an HDF5 file and visualizes it using plot_2d_counter.

    Args:
        h5_path (str): Path to the HDF5 file.
        group_name (str): Name of the group within the HDF5 file to extract data from.
        output_dir (str, optional): Directory to save the plot. If None, plot will be shown.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            group_name = yplus
            data_group = hf[group_name]
            datasets_to_plot = ['u', 'v','w']
            data_dict = {}
            first_valid_ds = None
            for ds_name in datasets_to_plot:
                if ds_name in data_group:
                    data_dict[ds_name] = data_group[ds_name][:]
                    if first_valid_ds is None:
                        first_valid_ds = data_dict[ds_name]
                else:
                    data_dict[ds_name] = None

            if first_valid_ds is None:
                print(f"错误: 在组 '{group_name}' 中未找到任何有效的数据集。")
                return

            print(f"成功从组 '{group_name}' 加载数据。")

            # --- 开始绘图 ---
            print("---开始绘图---")
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), constrained_layout=True)
            fig.suptitle(f'U V W Visualization\nFile: {os.path.basename(h5_path)} | Group: {group_name}',
                         fontsize=16)
            ax_flat = axes.flatten()

            for i, ds_name in enumerate(datasets_to_plot):
                ax = ax_flat[i]
                data = data_dict[ds_name]
                if data is None:
                    print("data is None")
                plot_2d_counter(data=data, mesh=mesh, ax=ax,title=ds_name)
            # --- 保存图像或显示 ---
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = f"{idx}_{yplus}_visualization.png"
                output_file_path = os.path.join(output_dir, filename)
                fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
                print(f"图像已成功保存至: {output_file_path}")
                plt.close(fig) # 关闭图表，释放内存
            else:
                plt.show() # 如果没有指定输出目录，则显示图表

    except FileNotFoundError:
        print(f"错误: HDF5 文件未找到: {h5_path}")
    except KeyError as e:
        print(f"错误: HDF5 文件结构不符合预期。键 '{e}' 不存在。")
    except Exception as e:
        print(f"绘图过程中发生未知错误: {e}")

def main():
    if os_name == 'nt':
        h5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
    else:
        h5_path = "/data_8T/Jinzun/HDF5/compressible_channel_flow_data_1490_1492.hdf5"
    pattern = '_(\d+)_(\d+)'
    idx = re.search(pattern,os.path.basename(h5_path)).group(1)

    output_dir = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science/output/input_visualization"
    y_plus_levels = ["yplus_wall_data", "yplus_1_data", "yplus_2_data", "yplus_5_data",
                     "yplus_10_data", "yplus_15_data", "yplus_30_data", "yplus_70_data",
                     "yplus_100_data", "yplus_200_data"]
    visualize_input_data(h5_path,yplus= y_plus_levels[3], idx=idx , output_dir=output_dir)


if __name__ == "__main__":
    main()
