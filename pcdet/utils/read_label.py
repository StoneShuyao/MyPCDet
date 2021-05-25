import numpy as np


def get_objects_from_label(label_file):
    with open(str(label_file), 'r') as f:
        lines = f.readlines()
    objects = [Label(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Label(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.box3d = np.array((float(label[1]), float(label[2]), float(label[3]), float(label[4]),
                               float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
