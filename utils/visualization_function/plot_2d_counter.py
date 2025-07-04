import numpy as np
import matplotlib.pyplot as plt

def plot_2d_counter(data, mesh, region=None, title=None, ax=None, formulate=None, vmin_fixed=None, vmax_fixed=None):
    """
    绘制计数器数据的二维截面图。

    参数:
    data (np.array): 要绘制的二维数据。
    mesh (tuple): 包含网格信息的元组，例如 (x_coords, y_coords, z_coords)。
    region (tuple, optional): (x_start, x_end, y_start, y_end) 用于指定数据的一个子区域。
    title (str, optional): 图表的标题。
    ax (matplotlib.axes.Axes, optional): 用于绘图的 Matplotlib 坐标轴对象。
    formulate (str, optional): 图表的备用标题。
    vmin_fixed (float, optional): 颜色条的固定最小值。如果为 None，则使用数据的最小值。
    vmax_fixed (float, optional): 颜色条的固定最大值。如果为 None，则使用数据的最大值。
    """

    # 如果没有提供 region，则使用整个数据范围
    if region is None:
        x_start, x_end = 0, data.shape[0]
        y_start, y_end = 0, data.shape[1]
    else:
        x_start, x_end, y_start, y_end = region
    ## -------------------------------------------------------------------------------------
    # 检查数据维度是否匹配
    if data.ndim != 2 or len(mesh[0]) != data.shape[0] or len(mesh[2]) != data.shape[1]:
        raise ValueError(f"无效错误: {data.ndim} {data.shape} != {len(mesh[0]), len(mesh[2])}")
    else:
        x_section = mesh[0][x_start:x_end]
        y_section = mesh[2][y_start:y_end]
        data_section = data[x_start:x_end, y_start:y_end]

        X, Y = np.meshgrid(x_section, y_section)

        # 确定 pcolormesh 的 vmin 和 vmax
        # 如果 vmin_fixed 不为空，则使用它，否则使用 data_section 的最小值
        vmin = vmin_fixed if vmin_fixed is not None else np.min(data_section)
        # 如果 vmax_fixed 不为空，则使用它，否则使用 data_section 的最大值
        vmax = vmax_fixed if vmax_fixed is not None else np.max(data_section)

        # 将 vmin 和 vmax 传递给 pcolormesh
        pcm = ax.pcolormesh(X, Y, data_section.T, cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)

    ## -------------------------------------------------------------------------------------
    # 添加颜色条并调整其位置
    fig = ax.get_figure()
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.016)

    # 设置颜色条的刻度位置和标签
    # 使用 pcolormesh 中实际使用的 vmin 和 vmax 来设置刻度位置
    tick_positions = np.linspace(vmin, vmax, 5)  # 生成5个刻度位置
    tick_labels = []
    for tick in tick_positions:
        # 格式化标签以保持一致的长度，并处理正负数对齐问题
        if tick >= 0:
            tick_labels.append(f" {tick:.3f}") # 正数前加一个空格用于对齐
        else:
            tick_labels.append(f"{tick:.3f}")

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    # 设置刻度值显示但不显示刻度线
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_tick_params(which='both', length=0)
    cbar.ax.yaxis.set_tick_params(which='both', length=0)

    ax.set_aspect('equal') # 保持 x 轴和 y 轴的单位长度相等，防止图形变形

    # 设置图表标题
    if formulate:
        ax.set_title(f"{formulate}")
    else:
        ax.set_title(f"{title}")

    return cbar