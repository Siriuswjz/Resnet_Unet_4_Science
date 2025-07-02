import numpy as np

def read_grid(grid_x, grid_y, grid_z):
    """
    读取网格文件STREAmS

    具体：读取第一列数据并转换为 numpy 数组。

    参数:
    grid_x, grid_y, grid_z (str): 网格文件路径，分别对应 x, y, z 方向的网格数据文件。

    返回:
    grid_data (list of numpy.ndarray): 包含 x, y, z 方向网格数据的列表。
    """

    def read_first_column(file_path):
        """
        读取文件的第一列数据。

        参数:
        file_path (str): 文件路径。

        返回:
        first_column_data (list of float): 第一列数据列表。
        """
        first_column_data = []
        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split()  # 假设数据由空格分隔
                if columns:  # 确保行不为空
                    # 尝试将字符串转换为浮点数
                    try:
                        value = float(columns[0])
                        first_column_data.append(value)
                    except ValueError:
                        print(f"Warning: Unable to convert {columns[0]} to float. Skipping this value.")
        return first_column_data

    # 读取三个文件的第一列数据，并存储到一个列表中
    grid_data = [
        read_first_column(grid_x),
        read_first_column(grid_y),
        read_first_column(grid_z)
    ]

    # 将列表转换为 numpy 数组，以便用于绘图
    grid_data = [np.array(data) for data in grid_data]

    return grid_data