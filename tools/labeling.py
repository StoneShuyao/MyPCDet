import argparse
import glob
from pathlib import Path
import re
import os
import subprocess

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visual_utils import visualize_utils as V


class LabelDataset(DatasetTemplate):
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


def cls_type_to_name(cls_type):
    type_to_name = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
    if cls_type not in type_to_name.keys():
        return -1
    return type_to_name[cls_type]


def write_label_file(pred_dicts, label_file):
    gt_types = pred_dicts[0]['pred_labels'].cpu().numpy()
    gt_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    gt_names = [cls_type_to_name(type_id) for type_id in gt_types]
    print("gt_boxes:", gt_boxes)
    print("scores:", scores)
    print("gt_names:", gt_names)

    with open(label_file, 'w') as f:
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
    return


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--label_path', type=str, default='data_label',
                        help='specify the data label file or directory')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    label_dataset = LabelDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(label_dataset)}')
    des_path = args.label_path
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=label_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(label_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            print(data_dict['frame_id'])
            sample_id = data_dict['frame_id']

            data_dict = label_dataset.collate_batch([data_dict])
            # print(data_dict['frame_id'])

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # print(pred_dicts)
            pcd_file = os.path.join(args.data_path, sample_id) + '.bin'
            label_file = os.path.join(des_path, sample_id) + '.txt'
            write_label_file(pred_dicts, label_file)
            subprocess.call(["xdg-open", label_file])
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=False)
            # subprocess.call(["xdg-open", label_file])

            flag = input("Input '1' to save the label, or will delete the bin and label file: ")
            if flag == '1':
                continue
            else:
                os.remove(pcd_file)
                os.remove(label_file)


    logger.info('Demo done.')


if __name__ == '__main__':
    main()
