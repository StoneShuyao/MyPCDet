import os
import random
import shutil
import fire


def add_new_data(data_path, new_path):
    """
    Add a set of newly collected data to the original/or a new dataset, the indexes
    of the frame files are follow the original indexes and  add one by one.

    Args:
        data_path: the new collected data path
        new_path: the dataset to be added

    Returns:

    """
    pcd_dir = os.path.join(data_path, 'lidar/')
    label_dir = os.path.join(data_path, 'label/')
    pcd_list = os.listdir(pcd_dir)
    label_list = os.listdir(label_dir)
    pcd_list.sort()
    label_list.sort()

    print("----Start to add %d frames data from %s to %s" % (len(pcd_list), data_path, new_path))

    if len(pcd_list) != len(label_list):
        assert print("num of pcd and label dismatch!")

    des_path = new_path

    idx_list = [file.split('.')[0] for file in pcd_list]
    # print(idx_list[0:10])

    if os.path.exists(des_path):
        lidar_path = os.path.join(des_path, 'lidar/')
        label_path = os.path.join(des_path, 'label/')
    else:
        os.makedirs(des_path)
        lidar_path = os.path.join(des_path, 'lidar/')
        os.makedirs(lidar_path)
        label_path = os.path.join(des_path, 'label/')
        os.makedirs(label_path)

    current_num = len(os.listdir(lidar_path))

    for idx in idx_list:
        shutil.copy((os.path.join(pcd_dir, idx)+'.bin'),
                    (os.path.join(lidar_path, "%06d" % current_num) + '.bin'))
        shutil.copy((os.path.join(label_dir, idx) + '.txt'),
                    (os.path.join(label_path, "%06d" % current_num) + '.txt'))
        current_num += 1
    print("----Finished adding data-----")


def shuffle_data(data_path):
    """
    To shuffle the frames in the dataset to make the adjacent frame are not continue
    in real world

    Args:
        data_path: The dataset to be shuffled, should contain both lidar and label

    Returns:

    """
    pcd_dir = os.path.join(data_path, 'lidar/')
    label_dir = os.path.join(data_path, 'label/')
    pcd_list = os.listdir(pcd_dir)
    label_list = os.listdir(label_dir)
    pcd_list.sort()
    label_list.sort()

    print("----Start to shuffle %d frames data in %s -----" % (len(pcd_list), data_path))

    if len(pcd_list) != len(label_list):
        assert print("num of pcd and label dismatch!")

    lidar_path = os.path.join(data_path, 'shuffled-lidar/')
    os.makedirs(lidar_path)
    label_path = os.path.join(data_path, 'shuffled-label/')
    os.makedirs(label_path)

    idx_list = [file.split('.')[0] for file in pcd_list]
    random.shuffle(idx_list)
    for num, idx in enumerate(idx_list):
        shutil.copy((os.path.join(pcd_dir, idx) + '.bin'),
                    (os.path.join(lidar_path, "%06d" % num) + '.bin'))
        shutil.copy((os.path.join(label_dir, idx) + '.txt'),
                    (os.path.join(label_path, "%06d" % num) + '.txt'))
    print("----Finished shuffle data-----")


if __name__ == "__main__":
    fire.Fire()
