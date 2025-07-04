from utils.visualization_function.visualize_h5_data import visualize_h5_data
import os
os_name = os.name

if __name__ == "__main__":
    if os_name == 'nt':
        h5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
        group_name = "yplus_wall_data"
        output_path = "D:/AI Codes/Resnet_Unet/output/truth_visualization"
        visualize_h5_data(h5_path, group_name, output_path)
    else:
        h5_path = "/data_8T/Jinzun/HDF5/compressible_channel_flow_data_1448_1450.hdf5"
        group_name = "yplus_wall_data"
        output_path = "/output/truth_visualization"
        visualize_h5_data(h5_path, group_name, output_path)
