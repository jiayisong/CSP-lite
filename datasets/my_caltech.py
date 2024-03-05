import os
import cv2
import numpy as np
from torchvision.datasets.vision import VisionDataset

from datasets import augmentations as augmentations


class CaltechDetection(VisionDataset):
    def __init__(self, root, image_set, input_size, down_factor, training):
        super(CaltechDetection, self).__init__(root, None, None, None)
        means = (103.5300, 116.2800, 123.6750)
        std = (57.3750, 57.1200, 58.3950)
        cls = 1
        self.avg_size = (0.1846, 0.0519)
        annotation_dir = os.path.join(root,
                                      'anno_train10x_alignedby_RotatedFilters' if image_set == 'train' else 'anno_test_1xnew')
        image_dir = os.path.join(root, 'frame')
        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset image not found or corrupted.')
        if not os.path.isdir(annotation_dir):
            raise RuntimeError('Dataset annotation not found or corrupted.')
        self.images = []
        self.annotations = []
        self.CLASSES = ('person',)
        self.training = training
        self.transforms = augmentations.Augmentation(input_size, means, std, down_factor, cls, training)
        self.name = 'caltech'
        '''
        self.test_config = {
            'Reasonable': {'height_range': (50, 1e10), 'visibility_range': (0.65, 1), 'overlap_threshold': 0.5,
                           'expanded_filtering': 1.25, 'bnds': (5, 5, 635, 475),
                           'FPPI_sample': (-2.0, 0.0)},
            'All': {'height_range': (20, 1e10), 'visibility_range': (0.2, 1), 'overlap_threshold': 0.5,
                    'expanded_filtering': 1.25, 'bnds': (5, 5, 635, 475),
                    'FPPI_sample': (-2.0, 0.0)},
            'Occ=heavy': {'height_range': (50, 1e10), 'visibility_range': (0.2, 0.65), 'overlap_threshold': 0.5,
                          'expanded_filtering': 1.25, 'bnds': (5, 5, 635, 475),
                          'FPPI_sample': (-2.0, 0.0)},
            'Scale=small': {'height_range': (50, 75), 'visibility_range': (0.65, 1),
                            'overlap_threshold': 0.5,
                            'expanded_filtering': 1.25, 'bnds': (5, 5, 635, 475),
                            'FPPI_sample': (-2.0, 0.0)
                            },
        }
        '''
        self.test_config = {
            'Reasonable': {'height_range': (50, 1e10), 'visibility_range': (0.65, 1), 'overlap_threshold': 0.5,
                           'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                           'FPPI_sample': (-2.0, 0.0)},
            'All': {'height_range': (20, 1e10), 'visibility_range': (0.2, 1), 'overlap_threshold': 0.5,
                    'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                    'FPPI_sample': (-2.0, 0.0)},
            'Occ=heavy': {'height_range': (50, 1e10), 'visibility_range': (0.2, 0.65), 'overlap_threshold': 0.5,
                          'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                          'FPPI_sample': (-2.0, 0.0)},
            'Scale=small': {'height_range': (50, 75), 'visibility_range': (0.65, 1),
                            'overlap_threshold': 0.5,
                            'expanded_filtering': 1.25, 'bnds': (-1e5, -1e5, 1e5, 1e5),
                            'FPPI_sample': (-2.0, 0.0)
                            }
        }

        self.gt_id = []
        self.no_gt_id = []
        ann = os.listdir(annotation_dir)
        ann.sort()
        for i, file in enumerate(ann):
            ann_root = os.path.join(annotation_dir, file)
            img_root = os.path.join(image_dir, (file[:-4] + '.jpg') if image_set == 'train' else file[:-4])
            if os.path.exists(img_root):
                self.annotations.append(ann_root)
                self.images.append(img_root)
            else:
                raise RuntimeError(img_root + ' img lost')
        self.sample_gt_id = []
        self.sample_no_gt_id = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = cv2.imread(self.images[index])
        if img is None:
            raise RuntimeError(self.images[index] + ' does not exist')
        if self.training:
            target = self.parse(self.annotations[index])
            return self.transforms(img, target)
        else:
            return self.transforms(img, None)


    def __len__(self):
        return len(self.images)

    def get_img(self, index):
        return cv2.imread(self.images[index])

    def get_img_name(self, index):
        return (((self.images[index].split('/'))[-1]).split('\\'))[-1]

    def parse(self, file, config='Reasonable'):
        label = []
        bbox = []
        bbox_viz = []
        is_difficult = []
        with open(file, 'r') as f:
            for line in f:
                if line[0] == '%':
                    continue
                # print(line,end='')
                splt = line.split()
                h = float(splt[4])
                w = float(splt[3])
                x_min = float(splt[1])
                y_min = float(splt[2])
                if self.training:
                    if w > 0 and h > 0:
                        if int(splt[10]):
                            difficult = True
                        else:
                            difficult = False
                            # if x_min + w / 2 < 0 or x_min + w / 2 > 639 or y_min + h / 2 < 0 or y_min + h / 2 > 479:
                            #    continue
                            # w = h * 0.41
                        is_difficult.append(difficult)
                        label.append(0)
                        bbox.append([x_min, y_min, x_min + w, y_min + h])
                        if int(splt[5]):
                            w_viz = float(splt[8])
                            h_viz = float(splt[9])
                            x_min_viz = float(splt[6])
                            y_min_viz = float(splt[7])
                            bbox_viz.append([max(0.0, x_min_viz), max(0.0, y_min_viz), min(639.0, x_min_viz + w_viz),
                                             min(479.0, y_min_viz + h_viz)])
                        else:
                            bbox_viz.append([max(0.0, x_min), max(0.0, y_min), min(639.0, x_min + w),
                                             min(479.0, y_min + h)])
                else:
                    visibility = (float(splt[8]) * float(splt[9])) / (h * w)
                    if int(splt[10]):
                        difficult = True
                    elif h > self.test_config[config]['height_range'][1] or h < \
                            self.test_config[config]['height_range'][0]:
                        difficult = True
                    elif int(splt[5]) and (visibility > self.test_config[config]['visibility_range'][
                        1] or visibility < self.test_config[config]['visibility_range'][0]):
                        difficult = True
                    elif y_min < self.test_config[config]['bnds'][1] or y_min + h > \
                            self.test_config[config]['bnds'][3] or x_min < \
                            self.test_config[config]['bnds'][0] or x_min + w > \
                            self.test_config[config]['bnds'][2]:
                        difficult = True
                    elif w / h < 0.40 or w / h > 0.42:
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
                    label.append(0)
        return bbox, bbox_viz, label, is_difficult

    def get_annotation(self, index, config):
        bbox, bbox_viz, label, is_difficult = self.parse(self.annotations[index], config)
        return {"height": 480, "width": 640, "boxes": np.array(bbox, dtype=np.float32),
                "labels": np.array(label, dtype=np.int64),
                "is_difficult": np.array(is_difficult, dtype=np.bool)}

    def set_train_size(self, size):
        self.transforms.set_train_size(size)

    def get_train_size(self):
        return self.transforms.Resize.size

    def save_result(self, index, boxes, scores, output_dir):
        filename = (self.images[index]).split('\\')[-1].split('/')[-1].split('_')
        set = filename[0]
        video = filename[1]
        frame_number = int(filename[2][1:6]) + 1
        output_dir = os.path.join(output_dir, 'valresults')
        set_path = os.path.join(output_dir, set)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
        video_path = os.path.join(set_path, video + '.txt')
        if len(boxes) > 0:
            f_res = np.repeat(frame_number, len(boxes), axis=0).reshape((-1, 1))
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]
            results = np.concatenate((f_res, boxes, scores[:, np.newaxis]), axis=-1)
            if os.path.isfile(video_path):
                old_results = np.loadtxt(video_path)
                if len(old_results.shape) == 1:
                    old_results = old_results[np.newaxis, :]
                results = np.concatenate((old_results, results), axis=0)
            np.savetxt(video_path, results, fmt='%6f')
