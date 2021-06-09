import os
import numpy as np
import fire
import open3d as o3d
import copy
import pcl


def joint_pcd(ref_file, align_file, matrix_file):
    ref_pcd = pcl.load_XYZI(ref_file).to_array()[:, :4]
    align_pcd = pcl.load_XYZI(align_file).to_array()[:, :4]
    t_matrix = np.loadtxt(matrix_file, dtype=np.float32)

    pcd_points = align_pcd[:, :3]
    ones = np.ones(pcd_points.shape[0])
    pcd_use = np.c_[pcd_points, ones]
    # print(align_pcd)
    # print(t_matrix)
    rotate_pcd_T = np.dot(t_matrix, pcd_use.T)
    rotate_pcd = rotate_pcd_T.T
    rotate_pcd = np.float32(rotate_pcd)

    aligned_pcd = np.c_[rotate_pcd[:, :3], align_pcd[:, 3]]

    joint_pcd = np.r_[ref_pcd, aligned_pcd]

    pcd_pcl = pcl.PointCloud_PointXYZI(joint_pcd)
    pcl.save(pcd_pcl, 'try.pcd', binary=True)


if __name__ == "__main__":
    fire.Fire()
