import glob

file_list = glob.glob("./data/NPZ/*/*.npz")
print(len(file_list))
