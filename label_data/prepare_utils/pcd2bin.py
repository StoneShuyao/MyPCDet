import os
import numpy as np
from numpy.core.numeric import zeros_like
import fire
import open3d as o3d
import copy

plane_threshold = 0.2
segment_interations = 50

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
    #print(axis_norm)
    axis_angle = -angle * axis_norm
    #print(axis_angle)
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

    # R1 = pcd.get_rotation_matrix_from_xyz((0, rotate_angle, 0))
    R1 = pcd.get_rotation_matrix_from_axis_angle(axis_angle)

    pcd_r.rotate(R1, center=(0, 0, 0))
    # R2 = mesh.get_rotation_matrix_from_xyz((0, 0, 2*np.pi/11))
    # mesh_r.rotate(R2, center=(0, 0, 0))
    plane_model1, inliers1 = pcd_r.segment_plane(distance_threshold=plane_threshold, ransac_n=3,
                                                 num_iterations=segment_interations)
    # print(plane_model1)
    pcd_final = copy.deepcopy(pcd_r).translate((0, 0, plane_model1[3] / plane_model[2] - 1.7))
    return pcd_final


def process_pcd(filepath):
    """
    First rotate the pcd to make the road plane parallel to XOY plane
    Add a zero column to make pcd format from (x,y,z) to (x,y,z,i)
    Args:
        filepath: pcd file

    Returns: processed point numpy array

    """
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_r = rotate2xy(pcd)
    lidar0 = np.asarray(pcd_r.points)
    lidar = np.c_[lidar0, np.zeros(lidar0.shape[0])]
    return np.array(lidar)
 
 
def convert(pcdfolder, binfolder):
    """
    The main function of the pcd to bin convert
    Args:
        pcdfolder: pcd files path
        binfolder: dest bin files path

    Returns:

    """
    num = 0
    current_path = os.getcwd()
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
        pl = process_pcd(pcd_file)
        pl = pl.reshape(-1, 4).astype(np.float32)
        pcd_file_new = os.path.join(des_path, "%06d" % num) + '.bin'
        pl.tofile(pcd_file_new)
        num += 1


if __name__ == "__main__":
    fire.Fire()
