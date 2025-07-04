from utils.visualization_function.visualize_h5_data import visualize_h5_data
import os
import re

os_name = os.name
pattern = '_(\d+)_(\d+)'
if __name__ == "__main__":

    if os_name == 'nt':
        h5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1448_1450.hdf5"
    else:
        h5_path = "/data_8T/Jinzun/HDF5/compressible_channel_flow_data_1448_1450.hdf5"

    idx = re.search(pattern,os.path.basename(h5_path)).group(1)
    output_path = "/output/truth_visualization"
    visualize_h5_data(h5_path, idx , output_path,vmin_fixed=0.142,vmax_fixed=0.240)
