import os
import fire
import random


def write_idx_file(set_list, set_file):
    with open(set_file, 'w') as f:
        for i, idx in enumerate(set_list):
            if i:
                f.write("\n%s" % idx)
            else:
                f.write("%s" % idx)


def split_dataset(data_path, image_sets, train_percentage):
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
    train_set = os.path.join(image_sets, 'train.txt')
    test_set = os.path.join(image_sets, 'val.txt')

    print("----Start to split %d frames data to %s ----" % (len(pcd_list), image_sets))

    idx_list = [file.split('.')[0] for file in pcd_list]
    random.shuffle(idx_list)

    frame_num = len(idx_list)
    print('Total frames number:', frame_num)
    train_num = round(frame_num * train_percentage)
    test_num = frame_num - train_num
    train_list = idx_list[:train_num]
    test_list = idx_list[train_num: frame_num]
    train_list.sort()
    test_list.sort()
    print('Train frames number: %d' % len(train_list))
    print('Test frames number: %d' % len(test_list))

    write_idx_file(train_list, train_set)
    write_idx_file(test_list, test_set)

    print("----Finished split data-----")


if __name__ == "__main__":
    fire.Fire()
