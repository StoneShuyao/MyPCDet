import argparse
import glob
from pathlib import Path
import re
import os

import mayavi.mlab as mlab
import numpy as np
import torch
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


# The threshold of the score of objects

class DetectionDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        sample_id = re.split(r'[/.]\s*', self.sample_file_list[index].strip())[-2]
        # sample_id = self.sample_file_list[index]
        # sample_id = self.sample_file_list[index].split('/')

        input_dict = {
            'points': points,
            'frame_id': sample_id,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def cls_type_to_name(cls_type):
    type_to_name = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
    if cls_type not in type_to_name.keys():
        return -1
    return type_to_name[cls_type]


def write_detection_files(pred_dicts, detection_file, car=0.5, pedestrian=0.5):
    """
    delete the objects with score lower than threshold of its class
    and write them into the detection result file
    Args:
        pred_dicts: detected object dicts
        detection_file: output file
        car: car threshold
        pedestrian: pedestrian threshold

    Returns:
        filtered pred_dicts
    """

    gt_types = pred_dicts[0]['pred_labels'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()

    if len(gt_types) == 0:
        print("No object detected!")
    else:
        current_idx = 0
        for idx in range(len(gt_types)):
            if gt_types[idx] == 1 and scores[idx] < car:
                del_flag = 1
            elif gt_types[idx] == 2 and scores[idx] < pedestrian:
                del_flag = 1
            else:
                del_flag = 0

            if del_flag:
                pred_dicts[0]['pred_labels'] = del_tensor_ele(pred_dicts[0]['pred_labels'], current_idx)
                pred_dicts[0]['pred_boxes'] = del_tensor_ele(pred_dicts[0]['pred_boxes'], current_idx)
                pred_dicts[0]['pred_scores'] = del_tensor_ele(pred_dicts[0]['pred_scores'], current_idx)
            else:
                current_idx += 1

    gt_types = pred_dicts[0]['pred_labels'].cpu().numpy()
    gt_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    gt_names = [cls_type_to_name(type_id) for type_id in gt_types]

    with open(detection_file, 'w') as f:
        for i in range(len(gt_names)):
            if i:
                f.write("\n%s %f %f %f %f %f %f %f %f"
                        % (gt_names[i], gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2],
                           gt_boxes[i][3], gt_boxes[i][4], gt_boxes[i][5],
                           gt_boxes[i][6], scores[i]))
            else:
                f.write("%s %f %f %f %f %f %f %f %f"
                        % (gt_names[i], gt_boxes[i][0], gt_boxes[i][1], gt_boxes[i][2],
                           gt_boxes[i][3], gt_boxes[i][4], gt_boxes[i][5],
                           gt_boxes[i][6], scores[i]))

    return True


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    # basic args
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/carla_models/lamppost_model/pointpillar_lamppost_local.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default='ckpt/pointpillars/lamppost/0121/checkpoint_epoch_70.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    # data and output path
    parser.add_argument('--single_seq', action='store_true', default=False,
                        help='Decide if data_path is single seq or multi seq in on folder')
    parser.add_argument('--data_path', type=str, default='detection_data',
                        help='specify the data folder need to be detected')

    # detection threshold
    parser.add_argument('--car_thres', type=float, default=0.5, help='confidence thres for vehicle detection')
    parser.add_argument('--ped_thres', type=float, default=0.5, help='confidence thres for pedestrian detection')

    # count frame
    parser.add_argument('--frame_num', type=int, default=150, help='detection time count frame number')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def detection_single_folder(args, cfg, logger, data_folder, detection_folder):
    detection_dataset = DetectionDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_folder), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(detection_dataset)}')

    if os.path.exists(detection_folder):
        pass
    else:
        os.makedirs(detection_folder)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=detection_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        time_list = []
        for idx, data_dict in enumerate(detection_dataset):
            sample_id = data_dict['frame_id']

            logger.info(f'Detecting sample index: \t{sample_id}')

            data_dict = detection_dataset.collate_batch([data_dict])
            # print(data_dict['frame_id'])

            load_data_to_gpu(data_dict)
            start_time = time.time()
            pred_dicts, _ = model.forward(data_dict)
            end_time = time.time()
            det_time = end_time - start_time
            time_list.append(det_time)

            # print(pred_dicts)
            detection_file = os.path.join(detection_folder, sample_id) + '.txt'
            # save_flag = write_detection_files(pred_dicts, detection_file,
            #                                   car=args.car_thres, pedestrian=args.ped_thres)
    return np.array(time_list)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Detecting of our Dataset by OpenPCDet-------------------------')

    collect_folder = args.data_path
    seq_list = os.listdir(collect_folder)
    frame_number = args.frame_num

    seq_time_list = np.zeros(frame_number)
    for seq in seq_list:
        seq_folder = os.path.join(collect_folder, seq)
        bin_folder = os.path.join(seq_folder, 'lamppost/lidar_bin')
        detection_folder = os.path.join(seq_folder, 'lamppost/detection')
        time_list = detection_single_folder(args, cfg, logger, bin_folder, detection_folder)
        if len(time_list) < frame_number:
            time_list = np.resize(time_list, (frame_number,))
        detection_frame_number = len(time_list)
        start_frame = int((detection_frame_number - frame_number) / 2)
        seq_time = time_list[start_frame: start_frame + frame_number]
        seq_time_list += seq_time

    seq_time_list = seq_time_list / len(seq_list)
    print('Mean Detection time', np.mean(seq_time_list))

    # save_result
    save_folder = "./results/overhead/detection/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = "0305_150_frame"

    save_path = os.path.join(save_folder, save_file)
    np.save(save_path, seq_time_list)

    logger.info('Detection done.')


if __name__ == '__main__':
    main()
