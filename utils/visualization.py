from utils.visualization_function.visualize_h5_data import visualize_h5_data
import os
import re

os_name = os.name
pattern = '_(\d+)_(\d+)'
if __name__ == "__main__":

    if os_name == 'nt':
        h5_path = "D:\AI Codes\Resnet_Unet\data\HDF5\compressible_channel_flow_data_1490_1492.hdf5"
    else:
        h5_path = "/data_8T/Jinzun/HDF5/compressible_channel_flow_data_1448_1450.hdf5"

    idx = re.search(pattern,os.path.basename(h5_path)).group(1)
    output_path = "/home/Jinzun/AI Codes/Resnet_Unet_4_Science/output/truth_visualization_extrema"
    extrema_dict = {'yplus_15_1490': [[-0.001,0.022],[-0.000,0.011],[0.152,0.254]],
                    'yplus_15_1448':[[-0.002,0.025],[-0.001,0.012],[0.142,0.240]]}
    visualize_h5_data(h5_path, idx , output_path,extrema=extrema_dict['yplus_15_1448'])
