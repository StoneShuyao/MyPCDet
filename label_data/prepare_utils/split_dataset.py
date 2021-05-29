import os
import fire


def split_dataset(data_path, image_sets):
    pcd_dir = os.path.join(data_path, 'lidar/')
    label_dir = os.path.join(data_path, 'label/')
    pcd_list = os.listdir(pcd_dir)
    label_list = os.listdir(label_dir)

    if len(pcd_list) != len(label_list):
        assert print("num of pcd and label dismatch!")

    if os.path.exists(image_sets):
        pass
    else:
        os.makedirs(image_sets)

    print("----Start to split %d frames data to %s ----" % (len(pcd_list), image_sets))


if __name__ == "__main__":
    fire.Fire()
