from utils.visualization_function.visualize_h5_data import visualize_h5_data

if __name__ == "__main__":
    h5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
    group_name = "yplus_wall_data"
    output_path = "D:/AI Codes/Resnet_Unet/output/visualization"
    visualize_h5_data(h5_path, group_name, output_path)