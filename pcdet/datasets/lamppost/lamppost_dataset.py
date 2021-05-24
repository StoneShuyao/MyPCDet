import copy
import pickle
import glob
from pathlib import Path

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti, read_label
from ..dataset import DatasetTemplate


class LamppostDataset(DatasetTemplate):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.ext = ext

        pc_file_list = glob.glob(str(self.root_split_path / 'lidar' / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        pc_file_list.sort()
        label_list = glob.glob(str(self.root_split_path / 'label' / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        label_list.sort()

        self.sample_file_list = pc_file_list
        self.label_file_list = label_list

        self.annos  = []
        self.get_annos()

    def get_label(self, idx):
        label_file = self.label_file_list[idx]
        assert label_file.exists()
        return read_label.get_objects_from_label(label_file)

    def get_annos(self):
        for index, label_file in enumerate(self.label_file_list):
            label_dict = {
                'frame_id': index,
            }
            obj_list = self.get_label(index)
            label_dict.update({
                'name': np.array([obj.cls_type for obj in obj_list]),
                'gt_boxes_lidar': np.concatenate([obj.box3d.reshape(1, 7) for obj in obj_list], axis=0)
            })
            self.annos.append(label_dict)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'gt_boxes_lidar' not in self.annos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Car': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = copy.deepcopy(self.annos)

        ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_file_list) * self.total_epochs

        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.sample_file_list)

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

        obj_list = self.get_label(index)
        input_dict.update({
            'gt_names': np.array([obj.cls_type for obj in obj_list]),
            'gt_boxes': np.concatenate([obj.box3d.reshape(1, 7) for obj in obj_list], axis=0)
        })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

