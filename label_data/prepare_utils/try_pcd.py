import pcl
import numpy as np


file = '/ssd/Data/Livox/lidar1/pcd/1/4157414.099341060.pcd'
pcd = pcl.load_XYZI(file).to_array()[:, :4]
intensity = pcd[:, 3]
pcd[:, 3] = intensity / 255
pcd = pcd.reshape(-1, 4).astype(np.float32)
pcd_file_new = '/ssd/Data/Livox/4157414.099341060.bin'
pcd.tofile(pcd_file_new)
print(intensity.max())
print(pcd)



