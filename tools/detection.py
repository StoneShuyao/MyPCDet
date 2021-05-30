import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


# The threshold of the score of objects
thres_car = 0.5
thres_pedestrian = 0.2
thres_cyclist = 0.3

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

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


def filter_pred_dicts(pred_dicts, car, pedestrian, cyclist):
    """
    delete the objects with score lower than threshold of its class
    Args:
        pred_dicts: detected object dicts
        car: car threshold
        pedestrian: pedestrian threshold
        cyclist: cyclist threshold

    Returns:
        filtered pred_dicts
    """

    gt_types = pred_dicts[0]['pred_labels'].cpu().numpy()
    gt_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
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
            elif gt_types[idx] == 3 and scores[idx] < cyclist:
                del_flag = 1
            else:
                del_flag = 0

            if del_flag:
                pred_dicts[0]['pred_labels'] = del_tensor_ele(pred_dicts[0]['pred_labels'], current_idx)
                pred_dicts[0]['pred_boxes'] = del_tensor_ele(pred_dicts[0]['pred_boxes'], current_idx)
                pred_dicts[0]['pred_scores'] = del_tensor_ele(pred_dicts[0]['pred_scores'], current_idx)
            else:
                current_idx += 1

    return pred_dicts


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Detect the current frame-------------------------')
    detection_dataset = DetectionDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(detection_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=detection_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(detection_dataset):

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = detection_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            print(pred_dicts)
            pred_dicts = filter_pred_dicts(pred_dicts=pred_dicts, car=thres_car,
                              pedestrian=thres_pedestrian, cyclist=thres_cyclist)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=False)

    logger.info('Detection done.')


if __name__ == '__main__':
    main()
