import os
import shutil
import fire


def number_files(file_folder, des_folder):
    """
    Number pcd files in single folder starting from 000000.pcd
    Args:
        file_folder: pcd files folder
        des_folder: destination folder

    Returns:

    """
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, file_folder)
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = os.path.join(current_path, des_folder)

    if os.path.exists(des_path):
        print('Error: folder already exist!')
        return
    else:
        os.makedirs(des_path)

    for idx, file_name in enumerate(file_list):
        shutil.copy(os.path.join(ori_path, file_name),
                    (os.path.join(des_path, "%06d" % idx) + '.pcd'))

    print('Done: number files in %s' % file_folder)

def number_all(total_folder, result_folder):
    """
    Batch number pcd files in folders
    Args:
        total_folder: folder contains pcd folders
        result_folder: destination folder

    Returns:

    """
    current_path = os.getcwd()
    total_path = os.path.join(current_path, total_folder)
    folder_list = os.listdir(total_path)
    folder_list.sort()
    result_path = os.path.join(current_path, result_folder)

    if os.path.exists(result_path):
        pass
    else:
        os.makedirs(result_path)

    for folder in folder_list:
        file_folder = os.path.join(total_path, folder)
        des_folder = os.path.join(result_path, folder)
        number_files(file_folder, des_folder)

    print('Done: number all files')


def number_dataset(ori_path, des_path):
    root_path = ori_path
    root_calib = os.path.join(root_path, 'calib')
    calib_list = os.listdir(root_calib)
    calib_list.sort()
    root_car = os.path.join(root_path, 'car_bin')
    car_list = os.listdir(root_car)
    car_list.sort()
    root_lamppost = os.path.join(root_path, 'lamppost_bin')
    lamppost_list = os.listdir(root_lamppost)
    lamppost_list.sort()
    root_joint = os.path.join(root_path, 'joint_bin')
    joint_list = os.listdir(root_joint)
    joint_list.sort()

    if os.path.exists(des_path):
        print('Dest path already exists!')
        return
    else:
        os.makedirs(des_path)
        des_calib = os.path.join(des_path, 'calib')
        os.makedirs(des_calib)
        des_car = os.path.join(des_path, 'car_bin')
        os.makedirs(des_car)
        des_lamppost = os.path.join(des_path, 'lamppost_bin')
        os.makedirs(des_lamppost)
        des_joint = os.path.join(des_path, 'joint_bin')
        os.makedirs(des_joint)

    for idx, sample_name in enumerate(calib_list):
        sample_id = sample_name.split('.')[0]
        if car_list[idx].split('.')[0] != sample_id:
            print('Sample ID %s does not match in Car list!')
            return
        elif lamppost_list[idx].split('.')[0] != sample_id:
            print('Sample ID %s does not match in Lamppost list!')
            return
        elif joint_list[idx].split('.')[0] != sample_id:
            print('Sample ID %s does not match in Joint list!')
            return
        else:
            calib_file0 = os.path.join(root_calib, calib_list[idx])
            car_file0 = os.path.join(root_car, car_list[idx])
            lamppost_file0 = os.path.join(root_lamppost, lamppost_list[idx])
            joint_file0 = os.path.join(root_joint, joint_list[idx])

            calib_file = os.path.join(des_calib, '%06d' % idx) + '.txt'
            car_file = os.path.join(des_car, '%06d' % idx) + '.bin'
            lamppost_file = os.path.join(des_lamppost, '%06d' % idx) + '.bin'
            joint_file = os.path.join(des_joint, '%06d' % idx) + '.bin'

            shutil.copy(calib_file0, calib_file)
            shutil.copy(car_file0, car_file)
            shutil.copy(lamppost_file0, lamppost_file)
            shutil.copy(joint_file0, joint_file)

    print('Finished number dataset %s' % ori_path)
    return


if __name__ == "__main__":
    fire.Fire()
