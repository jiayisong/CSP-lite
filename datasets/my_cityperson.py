import os
import cv2
import json
import numpy as np
from torchvision.datasets.vision import VisionDataset
from datasets import augmentations as augmentations


class CityPersonDetection(VisionDataset):
    def __init__(self, root, image_set, input_size, down_factor, training):
        super(CityPersonDetection, self).__init__(root, None, None, None)
        means = (103.5300, 116.2800, 123.6750)
        std = (57.3750, 57.1200, 58.3950)
        cls = 1
        # (2.0, 396.0, 46.24077211666465)
        # (6.0, 963.0, 114.00520392109404)
        self.avg_size = (0.1113, 0.0226)
        image_dir = os.path.join(root, 'leftImg8bit', image_set)
        annotation_dir = os.path.join(root, 'gtBboxCityPersons', image_set)
        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset image not found or corrupted.')
        if not os.path.isdir(annotation_dir):
            raise RuntimeError('Dataset annotation not found or corrupted.')
        self.images = []
        self.annotations = []
        self.parse_annotations = []
        self.training = training
        self.CLASSES = ('person',)
        self.name = 'cityperson'
        self.test_config = {
            'Reasonable': {'height_range': (50, 1e10), 'visibility_range': (0.65, 1), 'overlap_threshold': 0.5,
                           'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                           'FPPI_sample': (-2.0, 0.0)},
            'Large': {'height_range': (100, 1e10), 'visibility_range': (0.65, 1), 'overlap_threshold': 0.5,
                      'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                      'FPPI_sample': (-2.0, 0.0)},
            'Medium': {'height_range': (75, 100), 'visibility_range': (0.65, 1), 'overlap_threshold': 0.5,
                       'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                       'FPPI_sample': (-2.0, 0.0)},
            'Small': {'height_range': (50, 75), 'visibility_range': (0.65, 1),
                      'overlap_threshold': 0.5,
                      'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                      'FPPI_sample': (-2.0, 0.0)},
            'Bare': {'height_range': (50, 1e10), 'visibility_range': (0.9, 1),
                     'overlap_threshold': 0.5,
                     'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                     'FPPI_sample': (-2.0, 0.0)},
            'Partial': {'height_range': (50, 1e10), 'visibility_range': (0.65, 0.9),
                        'overlap_threshold': 0.5,
                        'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                        'FPPI_sample': (-2.0, 0.0)},
            'Heavy': {'height_range': (50, 1e10), 'visibility_range': (0, 0.65), 'overlap_threshold': 0.5,
                      'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                      'FPPI_sample': (-2.0, 0.0)},
        }

        citys = os.listdir(image_dir)
        citys.sort()
        for city in citys:
            imgs = os.listdir(os.path.join(image_dir, city))
            imgs.sort()
            for file in imgs:
                img_root = os.path.join(image_dir, city, file)
                ann_root = img_root.replace('leftImg8bit', 'gtBboxCityPersons').replace('.png', '.json')
                if os.path.exists(ann_root):
                    # if len(self.parse_json(ann_root)[2]) > 0 or (not training):
                    self.annotations.append(ann_root)
                    self.images.append(img_root)
                    self.parse_annotations.append(self.parse_json(ann_root))
                else:
                    raise RuntimeError(ann_root + ' annotations lost')

        self.transforms = augmentations.Augmentation(input_size, means, std, down_factor, cls, training)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        if self.training:
            img = cv2.imread(self.images[index])
            target = self.parse_annotations[index]
            return self.transforms(img, target)
        else:
            img = cv2.imread(self.images[index])
            return self.transforms(img, None)

    def __len__(self):
        return len(self.images)

    def get_img(self, index):
        return cv2.imread(self.images[index])

    def parse_json(self, json_file, config='Reasonable'):
        with open(json_file, 'r') as f:
            data = json.load(f)
        label = []
        bbox = []
        bbox_viz = []
        is_difficult = []
        for object in data['objects']:
            label.append(0)
            h = float(object['bbox'][3])
            w = float(object['bbox'][2])
            x_min = float(object['bbox'][0])
            y_min = float(object['bbox'][1])
            h_viz = float(object['bboxVis'][3])
            w_viz = float(object['bboxVis'][2])
            x_min_viz = float(object['bboxVis'][0])
            y_min_viz = float(object['bboxVis'][1])
            if self.training:
                if object['label'] == 'pedestrian':
                    is_difficult.append(False)
                else:
                    is_difficult.append(True)
                bbox.append([x_min, y_min, x_min + w, y_min + h])
                bbox_viz.append([max(0.0, x_min_viz), max(0.0, y_min_viz), min(2047.0, x_min_viz + w_viz),
                                 min(1023.0, y_min_viz + h_viz)])
                # if x_min_viz < 0 or y_min_viz < 0 or x_min_viz + w_viz > 2047 or y_min_viz + h_viz > 1023:
                #    print([x_min_viz, y_min_viz, x_min_viz + w_viz, y_min_viz + h_viz])
                #    print(json_file)
            else:
                visibility = (h_viz * w_viz) / (h * w)
                if object['label'] != 'pedestrian':
                    difficult = True
                elif h > self.test_config[config]['height_range'][1] or h < \
                        self.test_config[config]['height_range'][0]:
                    difficult = True
                elif (visibility > self.test_config[config]['visibility_range'][
                    1] or visibility < self.test_config[config]['visibility_range'][0]):
                    difficult = True
                elif y_min < self.test_config[config]['bnds'][1] or y_min + h > \
                        self.test_config[config]['bnds'][3] or x_min < \
                        self.test_config[config]['bnds'][0] or x_min + w > \
                        self.test_config[config]['bnds'][2]:
                    difficult = True
                else:
                    difficult = False
                is_difficult.append(difficult)
                '''
                if not difficult:
                    x_min = x_min + w * 0.5 - h * 0.205
                    w = h * 0.41
                '''
                bbox.append([x_min, y_min, x_min + w, y_min + h])
        return bbox, bbox_viz, label, is_difficult


    def get_img_name(self, index):
        return (((self.images[index].split('/'))[-1]).split('\\'))[-1]


    def get_annotation(self, index, config):
        json_file = self.annotations[index]
        with open(json_file, 'r') as f:
            data = json.load(f)
        bbox, _, label, is_difficult = self.parse_json(json_file, config)
        return {"height": data['imgHeight'], "width": data['imgWidth'], "boxes": np.array(bbox, dtype=np.float32),
                "labels": np.array(label, dtype=np.int64),
                "is_difficult": np.array(is_difficult, dtype=np.bool)}

    def set_train_size(self, size):
        self.transforms.set_train_size(size)

    def get_train_size(self):
        return self.transforms.Resize.size
