from dataclasses import replace

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def visualize_data(h5_path, group_name, output_path):
    """
    主函数，用于读取HDF5数据并绘图。
    """
    if not os.path.exists(h5_path):
        print(f"错误: HDF5文件未找到 -> {h5_path}")
        return

    print(f"正在从 '{h5_path}' 加载数据...")

    try:
        with (h5py.File(h5_path, 'r') as hf):
            data_group = hf[group_name]

            datasets_to_plot = ['u', 'v', 'w', 'pressure', 'friction_coefficient_2d', 'heat_flux_2d']
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

            # --- 核心修改：灵活处理坐标 ---
            physical_coords_found = False
            xlabel, ylabel = 'x (index)', 'z (index)'
            try:
                # 优先使用真实的物理坐标
                x = hf['x'][:]
                z = hf['z'][:]
                physical_coords_found = True
                xlabel, ylabel = 'x', 'z'
                print("已找到并使用 'x', 'z' 物理坐标。")
            except KeyError:
                # 如果找不到，则回退到使用数组索引
                print("警告: 未找到 'x', 'z' 坐标。将使用数组索引作为坐标轴。")
                # 从第一个加载的数据获取形状信息
                data_shape = first_valid_ds.shape
                x = np.arange(data_shape[0])
                z = np.arange(data_shape[1])

            print(f"成功从组 '{group_name}' 加载数据。")

            # --- 开始绘图 ---
            print("开始绘图...")
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 14), constrained_layout=True)
            fig.suptitle(f'Flow Field Visualization\nFile: {os.path.basename(h5_path)} | Group: {group_name}',
                         fontsize=16)
            ax_flat = axes.flatten()

            plot_titles = {
                'u': 'U Velocity', 'v': 'V Velocity', 'w': 'W Velocity',
                'p': 'Pressure (p)', 'cf': 'Skin Friction (Cf)', 'qw': 'Heat Flux (qw)'
            }

            for i, ds_name in enumerate(datasets_to_plot):
                ax = ax_flat[i]
                data = data_dict[ds_name]

                if data is None:
                    ax.text(0.5, 0.5, f"'{ds_name}'\nData not found", ha='center', va='center', style='italic',
                            color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                im = ax.pcolormesh(x, z, data.T, shading='auto', cmap='viridis')
                fig.colorbar(im, ax=ax)

                ax.set_title(plot_titles.get(ds_name, ds_name))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

                # 如果是物理坐标，保持正确的纵横比
                if physical_coords_found:
                    ax.set_aspect('equal', adjustable='box')

            # --- 保存图像 ---
            if output_path:
                filename = f"{group_name}_visualization.png"
                output_path = os.path.join(output_path, filename)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图像已成功保存至: {output_path}")
            plt.close(fig)

    except FileNotFoundError:
        print(f"错误: 文件 '{h5_path}' 不存在。")
    except KeyError:
        print(f"错误: 组 '{group_name}' 在文件 '{h5_path}' 中不存在。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == '__main__':
    hdf5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
    group_name = "yplus_wall_data"
    output_path = "D:/AI Codes/Resnet_Unet/output/result"
    visualize_data(hdf5_path, group_name, output_path)