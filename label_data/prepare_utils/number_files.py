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


if __name__ == "__main__":
    fire.Fire()
