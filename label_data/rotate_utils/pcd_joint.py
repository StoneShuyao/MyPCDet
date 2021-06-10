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
    # pcl.save(pcd_pcl, 'try.pcd', binary=True)
    return pcd_pcl


def joint_pcd_folder(ref_folder, align_folder, matrix_file, des_folder):
    ref_list = os.listdir(ref_folder)
    ref_list.sort()
    align_list = os.listdir(align_folder)
    align_list.sort()

    if os.path.exists(des_folder):
        print('%s already exist!' % des_folder)
        return
    else:
        os.makedirs(des_folder)

    for idx, file_name in enumerate(ref_list):
        if file_name != align_list[idx]:
            print('%s in %s dont have a match file in %s' % (file_name, ref_folder, align_folder))
            return
        else:
            ref_file = os.path.join(ref_folder, file_name)
            align_file = os.path.join(align_folder, file_name)
            joint_file = os.path.join(des_folder, file_name)
            joint_pcd_pcl = joint_pcd(ref_file, align_file, matrix_file)
            pcl.save(joint_pcd_pcl, joint_file, binary=True)

    print('Done: joint pcds in %s and %s' % (ref_folder, align_folder))


def joint_all(dataset_folder):
    root_path = dataset_folder
    ref_path = os.path.join(dataset_folder, 'car_use/')
    align_path = os.path.join(dataset_folder, 'lamppost_use/')
    matrix_path = os.path.join(dataset_folder, 'T_matrix/')
    des_path = os.path.join(dataset_folder, 'joint/')

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    ref_dir = os.listdir(ref_path)
    ref_dir.sort()
    align_dir = os.listdir(align_path)
    matrix_dir = os.listdir(matrix_path)

    for folder_name in ref_dir:
        if folder_name not in align_dir:
            print('%s dont have correspond align folder!' % folder_name)
            return

        matrix_name = folder_name + '.txt'
        if matrix_name not in matrix_dir:
            print('%s dont have correspond matrix file!' % folder_name)
            return

        ref_folder = os.path.join(ref_path, folder_name)
        align_folder = os.path.join(align_path, folder_name)
        matrix_file = os.path.join(matrix_path, matrix_name)
        des_folder = os.path.join(des_path, folder_name)

        joint_pcd_folder(ref_folder, align_folder, matrix_file, des_folder)

    print('Done: joint all pcd data in %s' % dataset_folder)
    return


if __name__ == "__main__":
    fire.Fire()
