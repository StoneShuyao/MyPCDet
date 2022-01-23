import numpy as np
import struct
import os
import sys
import open3d as o3d
import fire


def bin_to_pcd(binFileName):
    """
    the function to convert the .bin pointcloud file to pcd format
    Args:
        binFileName: bin file

    Returns:    open3d.pcd format (not a file)

    """
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def bin_to_pcd_folder(bin_folder, pcd_folder):
    """
    convert .bin files in a folder to pcd files and save in a new folder
    Args:
        bin_folder:
        pcd_folder:

    Returns:

    """
    if os.path.exists(pcd_folder):
        print('pcd_folder already exist!')
        return
    else:
        os.makedirs(pcd_folder)

    for bin_file in os.listdir(bin_folder):
        bin_path = os.path.join(bin_folder, bin_file)
        frame_id = bin_file.strip().split('.')[0]

        pcd = bin_to_pcd(bin_path)
        pcd_file = frame_id + '.pcd'
        pcd_path = os.path.join(pcd_folder, pcd_file)
        o3d.io.write_point_cloud(pcd_path, pcd)


if __name__ == "__main__":
    fire.Fire()
