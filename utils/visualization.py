import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_2d_counter(data, mesh, region=None, title=None, ax=None):
    """
    Plot a section of the counter data.(2D)
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    # 定义连续型颜色映射
    colors = ['#F0F0F0', '#E79A90', '#B02425']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    # 定义离散型颜色映射
    colors = ['#F0EFED', '#E79A90', '#B02425']
    cmap1 = ListedColormap(colors)
    ## -------------------------------------------------------------------------------------
    # 如果没有提供region，则使用整个数据范围
    if region is None:
        x_start, x_end = 0, data.shape[0]
        y_start, y_end = 0, data.shape[1]
    else:
        x_start, x_end, y_start, y_end = region
    ## -------------------------------------------------------------------------------------
    # 检查数据维度是否匹配
    if data.ndim != 2 or len(mesh[0]) != data.shape[0] or len(mesh[1]) != data.shape[1]:
        raise ValueError(f"Invalid error: {data.ndim} {data.shape} != {len(mesh[0]), len(mesh[1])}")
    else:
        x_section = mesh[0][x_start:x_end]
        y_section = mesh[1][y_start:y_end]
        data_section = data[x_start:x_end, y_start:y_end]
        X, Y = np.meshgrid(x_section, y_section)
        pcm = ax.pcolormesh(X, Y, data_section.T, cmap='coolwarm', shading='auto')
    ## -------------------------------------------------------------------------------------
    # 添加颜色条 调整位置
    fig = ax.get_figure()
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.016)  # ,label=f'{title} count colorbar')

    # 设置颜色条的刻度位置和标签
    vmin, vmax = np.min(data_section), np.max(data_section)
    tick_positions = np.linspace(vmin, vmax, 5)  # 生成5个刻度位置
    # 格式化为整数并统一长度修改位数
    tick_labels = []
    for tick in tick_positions:
        if tick >= 0:
            # 添加一个类似负号的隐藏标记，这里使用一个空格和减号组合来模拟
            tick_labels.append(f" {tick:.3f}")
        else:
            tick_labels.append(f"{tick:.3f}")
    # 特别处理负零的情况
    # tick_labels = [label if label != ("-0" or "-0.0" or "-0.00" )  else "0.00" for label in tick_labels]

    cbar.set_ticks(tick_positions)  # 设置刻度位置
    cbar.set_ticklabels(tick_labels)  # 设置刻度标签

    # 设置刻度值显示但不显示刻度线
    cbar.outline.set_visible(False)  # 隐藏颜色条的边框
    cbar.ax.xaxis.set_tick_params(which='both', length=0)
    cbar.ax.yaxis.set_tick_params(which='both', length=0)

    # # 使用FixedLocator来固定刻度的位置
    # cbar.locator = FixedLocator(cbar.get_ticks())
    # cbar.update_ticks()
    #
    # # 设置刻度标签为空字符串(那一条横线)
    # cbar.ax.set_xticklabels([''] * len(cbar.ax.get_xticks()))
    # # cbar.set_label('Colorbar Title', rotation=0, labelpad=15)

    ax.set_aspect('equal')  # 保持确保 x 轴和 y 轴的单位长度相等,防止图形变形:避免图形在绘制时被拉伸或压缩
    # # 设置图表标题,有传入标题这个参数,默认不显示设置成空字符串
    ax.set_title(f"{title}")

    return cbar

def visualize_h5_data(h5_path, group_name, output_dir=None):
    """
    Loads data from an HDF5 file and visualizes it using plot_2d_counter.

    Args:
        h5_path (str): Path to the HDF5 file.
        group_name (str): Name of the group within the HDF5 file to extract data from.
        output_dir (str, optional): Directory to save the plot. If None, plot will be shown.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            if group_name not in hf:
                print(f"错误: HDF5 文件中未找到组 '{group_name}'。")
                return

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

            try:
                # 优先使用真实的物理坐标
                x_coords = hf['x'][:]
                z_coords = hf['z'][:]
                physical_coords_found = True
                print("已找到并使用 'x', 'z' 物理坐标。")
            except KeyError:
                # 如果找不到，则回退到使用数组索引
                print("未找到 'x', 'z' 坐标。将使用数组索引作为坐标轴。")
                data_shape = first_valid_ds.shape
                x_coords = np.arange(data_shape[0])
                z_coords = np.arange(data_shape[1])

            mesh = (x_coords, z_coords) # 组织成 (x_array, y_array) 的元组

            print(f"成功从组 '{group_name}' 加载数据。")

            # --- 开始绘图 ---
            print("开始绘图...")
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), constrained_layout=True)
            fig.suptitle(f'Flow Field Visualization\nFile: {os.path.basename(h5_path)} | Group: {group_name}',
                         fontsize=16)
            ax_flat = axes.flatten()

            # 修正 plot_titles，使其键与 datasets_to_plot 中的字符串匹配
            plot_titles = {
                'friction_coefficient_2d': 'Skin Friction (Cf)',
                'heat_flux_2d': 'Heat Flux (qw)',
                'pressure': 'Pressure (p)'
            }

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

                # 调用 plot_2d_counter 函数
                plot_2d_counter(data=data, mesh=mesh, title=plot_titles.get(ds_name, ds_name), ax=ax)

            # --- 保存图像或显示 ---
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = f"{group_name}_visualization.png"
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

if __name__ == '__main__':
    hdf5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
    group_name = "yplus_wall_data"
    output_path = "D:/AI Codes/Resnet_Unet/output/result"
    visualize_h5_data(hdf5_path, group_name, output_path)