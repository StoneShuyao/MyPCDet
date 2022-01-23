import os
import random
import shutil
import fire


def add_carla_lp_data_all(all_path, new_path):
    """
    Summarize the multi lampposts data collected multiple times in this directory into one dataset at a time
    Args:
        all_path: path that contains multi lampposts data of multi times collection
        new_path:dataset path

    Returns:

    """
    collect_list = os.listdir(all_path)
    for collect_iter in collect_list:
        collect_path = os.path.join(all_path, collect_iter)
        lp_list = os.listdir(collect_path)
        for lp in lp_list:
            lp_path = os.path.join(collect_path, lp)
            data_path = os.path.join(lp_path, 'train_dataset')
            add_new_data(data_path, new_path)


def add_carla_vehicle_data_all(all_path, new_path):
    """
    Summarize the vehicle data collected multiple times in this directory into one dataset at a time
    Args:
        all_path: path that contains vehicle data of multi times collection
        new_path:

    Returns:

    """
    collect_list = os.listdir(all_path)
    for collect_iter in collect_list:
        collect_path = os.path.join(all_path, collect_iter)
        data_path = os.path.join(collect_path, 'train_dataset')
        add_new_data(data_path, new_path)


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


def filter_by_index(data_folder, index_folder, des_folder):
    """
    When change data to a new format but still want the same frames with original dataset
    Args:
        data_folder: data with new format
        index_folder: original format data
        des_folder: new data path

    Returns:

    """
    data_dir = data_folder
    idx_dir = index_folder
    idx_list = os.listdir(idx_dir)
    idx_list.sort()

    des_path = des_folder
    os.makedirs(des_path)

    for num, idx in enumerate(idx_list):
        data_file = os.path.join(data_dir, idx)
        des_file = os.path.join(des_path, idx)
        shutil.copy(data_file, des_file)


if __name__ == "__main__":
    fire.Fire()
