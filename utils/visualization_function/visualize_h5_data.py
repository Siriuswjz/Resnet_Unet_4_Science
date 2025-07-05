import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from utils.visualization_function.read_grid import read_grid
from utils.visualization_function.plot_2d_counter import plot_2d_counter

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
# 标题
plot_titles = {
    'friction_coefficient_2d': 'Skin Friction (Cf)',
    'heat_flux_2d': 'Heat Flux (Qw)',
    'pressure': 'Pressure (P)'
}
# 公式
plot_formulate = {'friction_coefficient_2d': r'$Cf = \frac{2 \tau_w}{\rho_b u_b^2}$',
                  'heat_flux_2d': r'$qw = \kappa \frac{\partial T}{\partial y}$'}

# 真实数据可视化绘图
def visualize_h5_data(h5_path, idx, output_dir=None,extrema=None):
    """
    Loads data from an HDF5 file and visualizes it using plot_2d_counter.

    Args:
        h5_path (str): Path to the HDF5 file.
        group_name (str): Name of the group within the HDF5 file to extract data from.
        output_dir (str, optional): Directory to save the plot. If None, plot will be shown.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            group_name = "yplus_wall_data"
            data_group = hf[group_name]
            datasets_to_plot = ['friction_coefficient_2d', 'heat_flux_2d','pressure']
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
            fig.suptitle(f'Flow Field Visualization\nFile: {os.path.basename(h5_path)} | Group: {group_name}',
                         fontsize=16)
            ax_flat = axes.flatten()

            for i, ds_name in enumerate(datasets_to_plot):
                ax = ax_flat[i]
                data = data_dict[ds_name]

                if data is None:
                    ax.text(0.5, 0.5, f"'{ds_name}'\nData not found", ha='center', va='center', style='italic',
                            color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(plot_titles.get(ds_name, ds_name)) # 即使数据不存在也设置标题
                    continue

                # 调用 plot_2d_counter 函数 从一个预定义的标题字典 plot_titles 中查找与数据集名称 ds_name 对应的更友好的标题。
                # 如果找不到（即 plot_titles 中没有 ds_name 这个键），它就直接使用 ds_name 作为标题。
                if extrema is not None:
                    plot_2d_counter(data=data, mesh=mesh, ax=ax,formulate=plot_formulate.get(ds_name, ds_name),
                                    vmin_fixed = extrema[i][0], vmax_fixed = extrema[i][1])
                else:
                    plot_2d_counter(data=data, mesh=mesh, ax=ax,formulate=plot_formulate.get(ds_name, ds_name))
            # --- 保存图像或显示 ---
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = f"{idx}_visualization.png"
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

# 模型预测绘图
def visualize_prediction_data(prediction_raw,idx,output_dir=None,extrema = None):
    datasets_to_plot = ['friction_coefficient_2d', 'heat_flux_2d', 'pressure']
    data_dict = {}
    for i,ds_name in enumerate(datasets_to_plot):
        data_dict[ds_name] = prediction_raw[i][:]
    print("---成功读取预测数据---")

    print("---开始绘图---")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), constrained_layout=True)
    fig.suptitle(f'Flow Field Visualization\n Group: yplus_wall_data',
                 fontsize=16)
    ax_flat = axes.flatten()

    for i, ds_name in enumerate(datasets_to_plot):
        ax = ax_flat[i]
        data = data_dict[ds_name]
        if data is None:
            ax.text(0.5, 0.5, f"'{ds_name}'\nData not found", ha='center', va='center', style='italic',
                    color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(plot_titles.get(ds_name, ds_name))  # 即使数据不存在也设置标题
            continue

        # 调用 plot_2d_counter 函数
        if extrema:
            plot_2d_counter(data=data, mesh=mesh, title=plot_titles.get(ds_name, ds_name),
                            ax=ax,vmin_fixed = extrema[i][0], vmax_fixed = extrema[i][1])
        else:
            plot_2d_counter(data=data, mesh=mesh, title=plot_titles.get(ds_name, ds_name),ax = ax)

    # --- 保存图像或显示 ---
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = f"{idx}_prediction_visualization.png"
        output_file_path = os.path.join(output_dir, filename)
        fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
        print(f"图像已成功保存至: {output_file_path}")
        plt.close(fig)  # 关闭图表，释放内存
    else:
        plt.show()  # 如果没有指定输出目录，则显示图表

# 误差分析绘图
def visualize_error_data(error_data,idx,output_dir=None,yplus=None):
    datasets_to_plot = ['Cf_error', 'Qw_error', 'P_error']
    data_dict = {}
    for i,ds_name in enumerate(datasets_to_plot):
        data_dict[ds_name] = error_data[i][:]
    print("---成功读取预测数据---")

    print("---开始绘图---")
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), constrained_layout=True)
    fig.suptitle(f'Error Visualization {yplus}',fontsize=16)
    ax_flat = axes.flatten()

    for i, ds_name in enumerate(datasets_to_plot):
        ax = ax_flat[i]
        data = data_dict[ds_name]
        if data is None:
            ax.text(0.5, 0.5, f"'{ds_name}'\nData not found", ha='center', va='center', style='italic',
                    color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(plot_titles.get(ds_name, ds_name))  # 即使数据不存在也设置标题
            continue
        plot_2d_counter(data=data, mesh=mesh, title = datasets_to_plot[i], ax = ax)

    # --- 保存图像或显示 ---
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = f"{idx}_prediction_visualization.png"
        output_file_path = os.path.join(output_dir, filename)
        fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
        print(f"图像已成功保存至: {output_file_path}")
        plt.close(fig)  # 关闭图表，释放内存
    else:
        plt.show()  # 如果没有指定输出目录，则显示图表





