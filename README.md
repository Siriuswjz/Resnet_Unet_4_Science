# train.py
1. prepare hdf5 data
2. use src/data/Hdf5_2_NPZ to get yplus layer feature and yplus wall target. you need fix layer 23 27 and 58
3. use utils/compute_feature_norm_std to compute the features' mean and std and copy it in the config
4. fix the layer 20 INPUT_Y_TYPE

# predict.py
