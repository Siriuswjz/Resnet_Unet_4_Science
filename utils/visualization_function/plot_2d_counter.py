import numpy as np
import matplotlib.pyplot as plt

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
    if data.ndim != 2 or len(mesh[0]) != data.shape[0] or len(mesh[2]) != data.shape[1]:
        raise ValueError(f"Invalid error: {data.ndim} {data.shape} != {len(mesh[0]), len(mesh[2])}")
    else:
        x_section = mesh[0][x_start:x_end]
        y_section = mesh[2][y_start:y_end]
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