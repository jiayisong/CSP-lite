import torch
import cv2
import math
import numpy as np
import random


class MakeHeatMap(object):
    def __init__(self, size, label_num, down_factor):
        self.size = size
        self.label_num = label_num
        self.down_factor = down_factor
        self.k_heat = 0.15
        self.k_bias = 0

    def __call__(self, boxes, boxes_viz, labels, is_difficult):
        size = np.array(self.size)
        heat_map_size = size // self.down_factor
        center_heatmap_pos = torch.zeros([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        center_heatmap_neg = torch.ones([self.label_num, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        size_mask = torch.zeros([1, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        size_heatmap = torch.ones([4, heat_map_size[0], heat_map_size[1]], dtype=torch.float32)
        k_heat = self.k_heat
        x = torch.arange(0, heat_map_size[1])
        x = x.unsqueeze(0).expand([heat_map_size[0], heat_map_size[1]])
        y = torch.arange(0, heat_map_size[0])
        y = y.unsqueeze(1).expand([heat_map_size[0], heat_map_size[1]])
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes_viz = boxes_viz.copy()
            boxes[:, ::2] = (boxes[:, ::2]) / self.down_factor
            boxes[:, 1::2] = (boxes[:, 1::2]) / self.down_factor
            boxes_viz[:, ::2] = (boxes_viz[:, ::2]) / self.down_factor
            boxes_viz[:, 1::2] = (boxes_viz[:, 1::2]) / self.down_factor
            boxes_ignore = boxes[is_difficult, :]
            labels_ignore = labels[is_difficult]
            for box, label in zip(boxes_ignore, labels_ignore):
                x1 = int(max(0, (box[0])))
                y1 = int(max(0, (box[1])))
                x2 = int(min((box[2]), heat_map_size[1] - 1))
                y2 = int(min((box[3]), heat_map_size[0] - 1))
                center_heatmap_neg[label, y1:y2 + 1, x1:x2 + 1] = 0
            boxes = boxes[~is_difficult, :]
            boxes_viz = boxes_viz[~is_difficult, :]
            labels = labels[~is_difficult]
            for box, box_viz, label in zip(boxes, boxes_viz, labels):
                center_x = (box[0] + box[2]) * 0.5
                center_y = (box[1] + box[3]) * 0.5
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                x1 = int(max(0, (box[0])))
                y1 = int(max(0, (box[1])))
                x2 = int(min((box[2]), heat_map_size[1] - 1))
                y2 = int(min((box[3]), heat_map_size[0] - 1))
                width = box[2] - box[0]
                height = box[3] - box[1]
                center_heatmap_temp = torch.exp(
                    -(((x[y1:y2 + 1, x1:x2 + 1] - center_x_int) / (k_heat * width + self.k_bias)) ** 2 + (
                            (y[y1:y2 + 1, x1:x2 + 1] - center_y_int) / (k_heat * height + self.k_bias)) ** 2) / 2)
                center_heatmap_neg[label, y1:y2 + 1, x1:x2 + 1] = torch.min(
                    center_heatmap_neg[label, y1:y2 + 1, x1:x2 + 1],
                    torch.pow(1 - center_heatmap_temp, 4))
                if center_x >= 0 and center_x <= heat_map_size[1] - 1 and center_y >= 0 and center_y <= heat_map_size[
                    0] - 1:
                    center_heatmap_neg[label, center_y_int, center_x_int] = 0
                    center_heatmap_pos[label, center_y_int, center_x_int] = 1
                    size_mask[0, center_y_int, center_x_int] = 1
                    size_heatmap[0, center_y_int, center_x_int] = float(width)
                    size_heatmap[1, center_y_int, center_x_int] = float(height)
                    size_heatmap[2, center_y_int, center_x_int] = center_x - center_x_int
                    size_heatmap[3, center_y_int, center_x_int] = center_y - center_y_int
        return center_heatmap_pos, center_heatmap_neg, size_heatmap, size_mask


class Augmentation(object):
    def __init__(self, size, mean, std, down_factor, cls, train):
        self.train = train
        self.SubtractMeans = SubtractMeans(mean, std)
        if train:
            self.PhotometricDistort = PhotometricDistort()
            self.RandomResize = RandomResize(size)
            self.RandomMirror = RandomMirror()
            self.ExpandOrCrop = ExpandOrCrop(size)
            self.MakeHeatMap = MakeHeatMap(size, cls, down_factor)
        else:
            self.Resize = Resize(size)

    def set_train_size(self, size):
        self.MakeHeatMap.size = size
        self.ExpandOrCrop.size = size

    def __call__(self, img, target):
        with torch.no_grad():
            if self.train:
                box, box_viz, label, is_difficult = target
                img = img.astype(np.float32)
                box = np.array(box, np.float32)
                box_viz = np.array(box_viz, np.float32)
                label = np.array(label, np.int32)
                is_difficult = np.array(is_difficult, np.bool8)
                img = self.PhotometricDistort(img)
                img = self.SubtractMeans(img)
                img, box, box_viz, label, is_difficult = self.RandomResize(img, box, box_viz, label, is_difficult)
                img, box, box_viz = self.RandomMirror(img, box, box_viz)
                img, box, box_viz, label, is_difficult = self.ExpandOrCrop(img, box, box_viz, label, is_difficult)

                heat_map = self.MakeHeatMap(box, box_viz, label, is_difficult)
                img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
                heat_map = list(heat_map)
                return [img] + heat_map
            else:
                img = img.astype(np.float32)
                img = self.SubtractMeans(img)
                img = self.Resize(img)
                img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
                return img



class Resize(object):
    def __init__(self, size):
        self.size = size
        assert size[0] % 32 == 0
        assert size[1] % 32 == 0

    def __call__(self, image):
        h, w, _ = image.shape
        H, W = self.size
        if H * w > W * h:
            scale = w / W
        else:
            scale = h / H
        new_w, new_h = int(w / scale), int(h / scale)
        if new_w == 0 or new_h == 0:
            raise RuntimeError('wrong img in dataset')
        resize_img = cv2.resize(image, (new_w, new_h))
        top, bottom, left, right = 0, H - new_h, 0, W - new_w
        image = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
        return image


class RandomResize(object):
    def __init__(self, size, lower=0.25, upper=1.75):
        self.size = size
        assert size[0] % 32 == 0
        assert size[1] % 32 == 0
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, boxes_viz, labels, is_difficult):
        h, w, _ = image.shape
        H, W = self.size
        while True:
            random_scale = random.uniform(self.lower, self.upper)
            if H * w > W * h:
                scale = w / W
            else:
                scale = h / H
            scale = scale / random_scale
            new_w, new_h = int(w / scale), int(h / scale)
            if new_w == 0 or new_h == 0:
                raise RuntimeError('wrong img in dataset')
            # print((new_w,new_h))
            resize_img = cv2.resize(image, (new_w, new_h))
            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] * new_w / w
                boxes[:, 2] = boxes[:, 2] * new_w / w
                boxes[:, 1] = boxes[:, 1] * new_h / h
                boxes[:, 3] = boxes[:, 3] * new_h / h
                boxes_viz[:, 0] = boxes_viz[:, 0] * new_w / w
                boxes_viz[:, 2] = boxes_viz[:, 2] * new_w / w
                boxes_viz[:, 1] = boxes_viz[:, 1] * new_h / h
                boxes_viz[:, 3] = boxes_viz[:, 3] * new_h / h
            return resize_img, boxes, boxes_viz, labels, is_difficult


class ExpandOrCrop(object):
    def __init__(self, size, object_prob=0):
        self.size = size
        self.object_prob = object_prob

    def crop_box(self, boxes, boxes_viz, labels, is_difficult, left, top, W, H):
        if len(boxes_viz) > 0:
            box = boxes.copy()
            box_viz = boxes_viz.copy()
            box_viz[:, :2] += (left, top)
            box_viz[:, 2:] += (left, top)
            box[:, :2] += (left, top)
            box[:, 2:] += (left, top)
            box_viz[:, 0::2] = np.clip(box_viz[:, 0::2], 0, W - 1)
            box_viz[:, 1::2] = np.clip(box_viz[:, 1::2], 0, H - 1)
            # centers = (box[:, :2] + box[:, 2:]) / 2.0
            # after_w = box[:, 2] - box[:, 0]
            after_h = box[:, 3] - box[:, 1]
            after_viz_w = box_viz[:, 2] - box_viz[:, 0]
            after_viz_h = box_viz[:, 3] - box_viz[:, 1]
            after_viz_area = after_viz_w * after_viz_h
            # after_area = after_w * after_h
            temp1 = after_viz_area > 1
            # temp2 = (0 <= centers[:, 0]) * (0 <= centers[:, 1]) * ((W - 1) >= centers[:, 0]) * ((H - 1) >= centers[:, 1])
            temp3 = (after_viz_w >= 16) * (after_viz_h >= 16)
            # temp3 = (after_viz_w * after_viz_h) >= 16*16
            # temp4 = after_viz_area > 0.16 * after_area
            # temp5 = after_h > self.min_h
            is_difficult = is_difficult + (~temp3)
            keep_object = temp1 * (~is_difficult)
            keep_difficult = temp1 * is_difficult
            keep_inds = keep_difficult + keep_object
            box_viz = box_viz[keep_inds, :]
            box = box[keep_inds, :]
            labels = labels[keep_inds]
            is_difficult = is_difficult[keep_inds]
            return box, box_viz, labels, is_difficult
        else:
            return boxes, boxes_viz, labels, is_difficult

    def crop_img(self, image, top, left, bottom, right, H, W):
        image = cv2.copyMakeBorder(image, max(top, 0), max(bottom, 0), max(left, 0), max(right, 0), cv2.BORDER_WRAP)
        image = image[max(0, -top):max(0, -top) + H, max(0, -left):max(0, -left) + W, :]
        return image

    def merge_boxes(self, bboxes):
        boxes, boxes_viz, labels, is_difficults = [], [], [], []
        for bbox in bboxes:
            box, box_viz, label, is_difficult = bbox
            boxes.append(box)
            boxes_viz.append(box_viz)
            labels.append(label)
            is_difficults.append(is_difficult)
        boxes = np.concatenate(boxes, axis=0)
        boxes_viz = np.concatenate(boxes_viz, axis=0)
        labels = np.concatenate(labels, axis=0)
        is_difficults = np.concatenate(is_difficults, axis=0)
        return boxes, boxes_viz, labels, is_difficults

    def move_boxes(self, bb, left, top):
        boxes, boxes_viz, labels, is_difficults = bb
        box = boxes.copy()
        box_viz = boxes_viz.copy()
        label = labels.copy()
        is_difficult = is_difficults.copy()
        if len(is_difficults) > 0:
            box_viz[:, :2] += (left, top)
            box_viz[:, 2:] += (left, top)
            box[:, :2] += (left, top)
            box[:, 2:] += (left, top)
        return box, box_viz, label, is_difficult

    def __call__(self, image, boxes, boxes_viz, labels, is_difficult):
        height, width, depth = image.shape
        H, W = self.size
        if (~is_difficult).sum() > 0 and random.uniform(0, 1) < self.object_prob:
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            gt_id = np.nonzero(
                (~is_difficult) * (0 <= centers[:, 0]) * (0 <= centers[:, 1]) * ((width - 1) >= centers[:, 0]) * (
                        (height - 1) >= centers[:, 1]))
            gt_id = gt_id[0]
            if len(gt_id) > 0:
                lucky_gt = random.choice(gt_id)
                x_min, y_min, x_max, y_max = boxes[lucky_gt, :]
                x_center, y_center = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
                top = -random.randint(max(1 - H // 2, math.ceil(y_center) - H), min(height - H // 2, int(y_center)))
                left = -random.randint(max(1 - W // 2, math.ceil(x_center) - W), min(width - W // 2, int(x_center)))
            else:
                top = random.randint(H // 2 - height, H // 2 - 1)
                left = random.randint(W // 2 - width, W // 2 - 1)
        else:
            top = random.randint(H // 2 - height, H // 2 - 1)
            left = random.randint(W // 2 - width, W // 2 - 1)
        right = W - width - left
        bottom = H - height - top
        image = self.crop_img(image, top, left, bottom, right, H, W)
        bb1 = boxes, boxes_viz, labels, is_difficult
        top_repeat_num = top // height + 1
        bottom_repeat_num = bottom // height + 1
        bb = []
        for i in range(top_repeat_num + bottom_repeat_num + 1):
            bb.append(self.move_boxes(bb1, 0, height * i))
        bb1 = self.merge_boxes(bb)
        left_repeat_num = left // width + 1
        right_repeat_num = right // width + 1
        bb = []
        for i in range(left_repeat_num + right_repeat_num + 1):
            bb.append(self.move_boxes(bb1, width * i, 0))
        boxes, boxes_viz, labels, is_difficult = self.merge_boxes(bb)
        top = top - height * top_repeat_num
        left = left - width * left_repeat_num
        boxes, boxes_viz, labels, is_difficult = self.crop_box(boxes, boxes_viz, labels, is_difficult, left, top, W, H)
        return image, boxes, boxes_viz, labels, is_difficult


class RandomMirror(object):
    def __call__(self, image, boxes, boxes_viz):
        _, width, _ = image.shape
        if random.randint(0, 1):
            image = image[:, ::-1]
            if len(boxes) > 0:
                boxes[:, 0::2] = width - 1 - boxes[:, 2::-2]
                boxes_viz[:, 0::2] = width - 1 - boxes_viz[:, 2::-2]
        return image, boxes, boxes_viz


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=2.0):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(0, 1):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            image[:, :, 1] = np.minimum(image[:, :, 1], 1)
        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(0, 1):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class RandomValue(object):
    def __init__(self, lower=0.5, upper=2.0):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."

    def __call__(self, image):
        if random.randint(0, 1):
            image[:, :, 2] *= random.uniform(self.lower, self.upper)
            image[:, :, 2] = np.minimum(image[:, :, 2], 255)
        return image


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(0, 1):
            swap = self.perms[random.randint(0, len(self.perms) - 1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            # RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            RandomValue(),
            ConvertColor(current='HSV', transform='BGR'),
            # RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        '''
        im = self.rand_brightness(im)
        if random.randint(0, 1):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        '''
        distort = Compose(self.pd)
        image = distort(image)
        # im, boxes, labels, is_difficult = self.rand_light_noise(im, boxes, labels, is_difficult)
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class SubtractMeans(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        image -= self.mean
        image /= self.std
        return image
