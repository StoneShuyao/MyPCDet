import os
import numpy as np
import fire
import open3d as o3d
import copy
import pcl
import shutil


plane_threshold = 0.2
segment_interations = 150


def getAngle(plane_model):
    plane_norm = np.array(plane_model[0:3])
    xy_norm = np.array([0,0,1])
    Lplane=np.sqrt(plane_norm.dot(plane_norm))
    Lxy=np.sqrt(xy_norm.dot(xy_norm))
    cos_angle=plane_norm.dot(xy_norm)/(Lplane*Lxy)
    angle=np.arccos(cos_angle)
    return angle


def getAxisAngle(plane_model):
    plane_norm = np.array(plane_model[0:3])
    xy_norm = np.array([0,0,1])
    Lplane=np.sqrt(plane_norm.dot(plane_norm))
    Lxy=np.sqrt(xy_norm.dot(xy_norm))
    cos_angle=plane_norm.dot(xy_norm)/(Lplane*Lxy)
    angle=np.arccos(cos_angle)

    axis = np.cross(xy_norm, plane_norm)
    axis_norm = axis/np.linalg.norm(axis)
    # print(axis_norm)
    axis_angle = -angle * axis_norm
    # print(axis_angle)
    return axis_angle


def rotate2xy(pcd):
    """
    Rotate the road plane to XOY Plane
    Args:
        pcd: open3d pointcloud

    Returns: rotated open3d pointcloud

    """
    pcd_r = copy.deepcopy(pcd)

    # pcd_r.paint_uniform_color([1, 0.706, 0])
    plane_model, inliers = pcd.segment_plane(distance_threshold=plane_threshold, ransac_n=3,
                                             num_iterations=segment_interations)
    rotate_angle = getAngle(plane_model)
    axis_angle = getAxisAngle(plane_model)

    # print(plane_model[3] / plane_model[2])

    # R1 = pcd.get_rotation_matrix_from_xyz((0, rotate_angle, 0))
    R1 = pcd.get_rotation_matrix_from_axis_angle(axis_angle)

    pcd_r.rotate(R1, center=(0, 0, 0))
    # R2 = mesh.get_rotation_matrix_from_xyz((0, 0, 2*np.pi/11))
    # mesh_r.rotate(R2, center=(0, 0, 0))
    plane_model1, inliers1 = pcd_r.segment_plane(distance_threshold=plane_threshold, ransac_n=3,
                                                 num_iterations=segment_interations)
    # print(plane_model1)
    pcd_final = copy.deepcopy(pcd_r).translate((0, 0, plane_model1[3] / plane_model1[2] - 1.7))
    return pcd_final


def process_pcd(filepath):
    """
    Add a zero column to make pcd format from (x,y,z) to (x,y,z,i)
    Args:
        filepath: pcd file

    Returns: processed point numpy array

    """
    pcd = o3d.io.read_point_cloud(filepath)
    lidar0 = np.asarray(pcd.points)
    lidar = np.c_[lidar0, np.zeros(lidar0.shape[0])]
    return np.array(lidar)


def rotate_pcd(pcdfolder, rotatedfolder, rootpath='./'):
    """
    Rotate the pcd to make the road plane parallel to XOY plane,
    and make the viewpoint be 1.7m high
    Args:
        pcdfolder: pcd files
        rotatedfolder: rotated pcd files path
        rootpath:

    Returns:

    """
    cur_path = rootpath
    ori_path = os.path.join(cur_path, pcdfolder)
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = os.path.join(cur_path, rotatedfolder)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    num = len(os.listdir(des_path))

    for file in file_list:
        (filename, extension) = os.path.splitext(file)
        pcd_file = os.path.join(ori_path, filename) + '.pcd'
        pcd_4array = pcl.load_XYZI(pcd_file).to_array()[:, :4]
        pcd_0 = o3d.io.read_point_cloud(pcd_file)
        pcd_r = rotate2xy(pcd_0)
        pcd_3array = np.array(pcd_r.points)
        pcd_4array[:, :3] = pcd_3array
        pcd_pcl = pcl.PointCloud_PointXYZI(pcd_4array)
        pcd_file_new = os.path.join(des_path, "%06d" % num) + '.pcd'
        # o3d.io.write_point_cloud(pcd_file_new, pcd_r)
        pcl.save(pcd_pcl, pcd_file_new, binary=True)

        num += 1
 
 
def convert(pcdfolder, binfolder, rootpath='./'):
    """
    The main function of the pcd to bin convert
    Args:
        pcdfolder: pcd files path
        binfolder: dest bin files path
        rootpath:

    Returns:

    """
    current_path = rootpath
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = os.path.join(current_path, binfolder)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    num = len(os.listdir(des_path))

    for file in file_list: 
        (filename, extension) = os.path.splitext(file)
        pcd_file = os.path.join(ori_path, filename) + '.pcd'
        # pl = process_pcd(pcd_file)
        # pl = pl.reshape(-1, 4).astype(np.float32)
        pl = pcl.load_XYZI(pcd_file).to_array()[:, :4]

        intensity = pl[:, 3]
        pl[:, 3] = intensity / 255    # normalize the intensity

        pl = pl.reshape(-1, 4).astype(np.float32)
        pcd_file_new = os.path.join(des_path, "%06d" % num) + '.bin'
        pl.tofile(pcd_file_new)
        num += 1


def convert_with_matrix(pcdfolder, matrix_folder, binfolder, matrix_dir, rootpath='./'):
    """
    The main function of the pcd to bin convert
    Args:
        pcdfolder: pcd files path
        matrix_folder: transform matrix folder
        binfolder: dest bin files path
        matrix_dir: folder of matrix one by one
        rootpath:

    Returns:

    """
    current_path = rootpath
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = os.path.join(current_path, binfolder)

    matrix_file_name = pcdfolder + '.txt'

    matrix_path = os.path.join(current_path, matrix_folder)
    matrix_list = os.listdir(matrix_path)
    matrix_list.sort()
    matrix_des_path = os.path.join(current_path, matrix_dir)

    if matrix_file_name not in matrix_list:
        print('Error! %s not in matrix list' % matrix_file_name)
        return
    else:
        matrix_to_copy = os.path.join(matrix_path, matrix_file_name)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    if os.path.exists(matrix_des_path):
        pass
    else:
        os.makedirs(matrix_des_path)

    num = len(os.listdir(des_path))

    for file in file_list:
        (filename, extension) = os.path.splitext(file)
        pcd_file = os.path.join(ori_path, filename) + '.pcd'
        # pl = process_pcd(pcd_file)
        # pl = pl.reshape(-1, 4).astype(np.float32)
        pl = pcl.load_XYZI(pcd_file).to_array()[:, :4]

        intensity = pl[:, 3]
        pl[:, 3] = intensity / 255    # normalize the intensity

        pl = pl.reshape(-1, 4).astype(np.float32)
        pcd_file_new = os.path.join(des_path, "%06d" % num) + '.bin'
        pl.tofile(pcd_file_new)

        matrix_des_file = os.path.join(matrix_des_path, "%06d" % num) + '.txt'
        shutil.copy(matrix_to_copy, matrix_des_file)
        num += 1


def rotate_all(pcdfolder, rotate_folder):
    """
    Batch rotate pcd in folders
    Args:
        pcdfolder: folder contains pcd folders
        rotate_folder: bin folder

    Returns:

    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    folder_list = os.listdir(ori_path)
    folder_list.sort()
    des_path = os.path.join(current_path, rotate_folder)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    for folder in folder_list:
        des_folder = os.path.join(rotate_folder, folder)
        rotate_pcd(folder, des_folder, rootpath=pcdfolder)
        print('rotating %s' % folder)


def convert_all(pcdfolder, binfolder):
    """
    Batch convert pcd in folders into bin
    Args:
        pcdfolder: folder contains pcd folders
        binfolder: bin folder

    Returns:

    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    folder_list = os.listdir(ori_path)
    folder_list.sort()
    des_path = os.path.join(current_path, binfolder)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    for folder in folder_list:
        convert(folder, binfolder, rootpath=pcdfolder)
        print('Converting %s to bin' %folder)


def convert_all_with_matrix(pcdfolder, matrix_folder, binfolder, matrix_dir):
    """
    Batch convert pcd in folders into bin
    Args:
        pcdfolder: folder contains pcd folders
        matrix_folder: folder contains matrix
        binfolder: bin folder
        matrix_dir: matrix one by one

    Returns:

    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    folder_list = os.listdir(ori_path)
    folder_list.sort()
    des_path = os.path.join(current_path, binfolder)

    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    for folder in folder_list:
        convert_with_matrix(folder, matrix_folder, binfolder, matrix_dir, rootpath=pcdfolder)
        print('Converting %s to bin' % folder)


if __name__ == "__main__":
    fire.Fire()
