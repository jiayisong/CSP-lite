import cv2
import os
from datetime import datetime
from collections import defaultdict
import itertools
import numpy as np


def resize(boxes, test_size, down_factor, w, h):
    boxes = boxes.copy()
    H, W = test_size
    x_bias = 0
    y_bias = 0
    boxes[:, ::2] = boxes[:, ::2] * down_factor + x_bias
    boxes[:, 1::2] = boxes[:, 1::2] * down_factor + y_bias
    if H * w > W * h:
        scale = w / W
    else:
        scale = h / H
    new_w, new_h = int(w / scale), int(h / scale)
    top, bottom, left, right = 0, H - new_h, 0, W - new_w
    boxes[:, ::2] = np.clip((boxes[:, ::2] - left) * (w * 1.0 / new_w), 0, w - 1)
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - top) * (h * 1.0 / new_h), 0, h - 1)
    # boxes[:, ::2] = (boxes[:, ::2] - left) * (w * 1.0 / new_w)
    # boxes[:, 1::2] = (boxes[:, 1::2] - top) * (h * 1.0 / new_h)
    return boxes


def filtering_detections(prediction, h_min, h_max):
    h = prediction['resize_boxes'][:, 3] - prediction['resize_boxes'][:, 1]
    ind = (h >= h_min) * (h < h_max)
    boxes = prediction['resize_boxes'][ind, :]
    labels = prediction['labels'][ind]
    scores = prediction['scores'][ind]
    return boxes, labels, scores


def evaluation(dataset, predictions, output_dir, test_size, down_factor, iteration=None, debug=None):
    class_names = dataset.CLASSES
    result_str = ""
    output_dir = output_dir + 'mMR/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if iteration is not None:
        print('\n' + "%s:BigStep[%d] " % (datetime.now(), iteration), end='')
        result_path = os.path.join(output_dir, 'result_{:03d}.txt'.format(iteration))
    else:
        print('\n' + "%s:" % (datetime.now()), end='')
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    res = []
    for config in dataset.test_config.keys():
        result_str = result_str + config + '\n'
        result = evaluation_a_config(dataset, predictions, test_size, down_factor, debug, config, output_dir)
        result_str = result_str + "mMR: {:.4f}\n".format(result["mmr"])
        metrics = {'mMR': result["mmr"]}
        for i, ap in enumerate(result["mr"]):
            metrics[class_names[i]] = ap
            result_str = result_str + "{:<16}: {:.4f}\n".format(class_names[i], ap)
        print("%s mMR:%f " % (config, result["mmr"]), end='')
        res.append(result["mmr"])
        #break
    print('')
    with open(result_path, "w") as f:
        f.write(result_str)
    return res


def evaluation_a_config(dataset, predictions, test_size, down_factor, debug, config, output_dir):
    class_names = dataset.CLASSES
    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []
    # res = []
    output_dir = output_dir + 'image/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if debug:
        colors = []
        for _ in range(len(class_names)):
            color = np.random.random((3,)) * 0.6 + 0.4
            color = color * 255
            color = color.astype(np.int32).tolist()
            colors.append(color)
    for i in range(len(dataset)):
        annotation = dataset.get_annotation(i, config)
        gt_boxes_list.append(annotation['boxes'])
        gt_labels_list.append(annotation['labels'])
        gt_difficults.append(annotation['is_difficult'])
        prediction = predictions[i]
        prediction['resize_boxes'] = resize(prediction['boxes'], test_size, down_factor, annotation['width'],
                                            annotation['height'])
        boxes, labels, scores = filtering_detections(prediction, dataset.test_config[config]['height_range'][
            0] / dataset.test_config[config]['expanded_filtering'], dataset.test_config[config]['height_range'][1] *
                                                     dataset.test_config[config]['expanded_filtering'])
        # dataset.save_result(i, np.array(boxes), np.array(scores), './summary/')
        # dataset.save_result(i, np.array(annotation['boxes']), np.array(annotation['is_difficult']), './summary/')
        '''
        if len(boxes) > 0:
            for j, box in enumerate(boxes):
                # box[:4] = box[:4] / 0.6
                box = box.copy()
                box[[2, 3]] -= box[[0, 1]]
                temp = dict()
                temp['image_id'] = i + 1
                temp['category_id'] = 1
                temp['bbox'] = box.tolist()
                temp['score'] = float(scores[j])
                res.append(temp)
        '''
        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)
        if debug:
            img = dataset.get_img(i)
            gt_boxes, gt_labels = annotation['boxes'], annotation['labels']
            img_pre = img.copy()
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                gt_box = gt_box.copy().astype(np.int32)
                cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), colors[gt_label], 2)
                cv2.putText(img, class_names[gt_label], (gt_box[0] + 2, gt_box[1] + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[gt_label], 1)
            #cv2.imshow("target", img)
            j = 0
            if scores.size != 0:
                while scores[j] > debug:
                    # for _ in range(5):
                    boxe = boxes[j].copy().astype(np.int32)
                    cv2.rectangle(img_pre, (boxe[0], boxe[1]), (boxe[2], boxe[3]), colors[labels[j]], 2)
                    # cv2.putText(img_pre, class_names[labels[j]] + ':' + "{:.2f}".format(scores[j]),(boxe[0] + 2, boxe[1] + 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[labels[j]], 1)
                    cv2.putText(img_pre, "{:.2f}".format(scores[j]), (boxe[0] + 2, boxe[1] + 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, colors[labels[j]], 1)
                    j += 1
                    if j >= scores.size:
                        break
            #cv2.imshow("prediction", img_pre)
            #cv2.waitKey(0)
            cv2.imwrite(os.path.join(output_dir, dataset.get_img_name(i)), img_pre)

    # with open('./summary/cityperson_result.json', 'w') as f:
    #    json.dump(res, f)

    result = eval_detection(pred_bboxes=pred_boxes_list, pred_labels=pred_labels_list, pred_scores=pred_scores_list,
                            gt_bboxes=gt_boxes_list, gt_labels=gt_labels_list, gt_difficults=gt_difficults,
                            iou_thresh=dataset.test_config[config]['overlap_threshold'], use_07_metric=True,
                            FPPI_sample=dataset.test_config[config]['FPPI_sample'])

    return result


def bbox_iou(bbox_a, bbox_b, difficult):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b * (~difficult) - area_i * (~difficult))


def eval_detection(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh, gt_difficults,
                   use_07_metric, FPPI_sample):
    prec, rec, miss_rate, false_positive_per_image = calc_detection_prec_rec(pred_bboxes, pred_labels, pred_scores,
                                                                             gt_bboxes, gt_labels, gt_difficults,
                                                                             iou_thresh=iou_thresh)
    draw_p_r = False
    if draw_p_r:
        import matplotlib.pyplot as mpt
        for p, r in zip(prec, rec):
            mpt.figure()
            mpt.plot(r, p)
            mpt.grid()
        mpt.show()
    draw_FPPI = False
    if draw_FPPI:
        import matplotlib.pyplot as mpt
        for p, r in zip(miss_rate, false_positive_per_image):
            mpt.figure()
            mpt.plot(r, p)
            mpt.yscale("log")
            mpt.xscale("log")
            mpt.grid()
        mpt.show()
    ap = calc_detection_ap(prec, rec, use_07_metric=use_07_metric)
    mr = calc_detection_mr(miss_rate, false_positive_per_image, FPPI_sample)
    return {'ap': ap, 'map': np.nanmean(ap), 'mr': mr, 'mmr': np.nanmean(mr)}


def calc_detection_prec_rec(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
                            iou_thresh=0.5):
    N = len(gt_labels)
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)
    n_pos = defaultdict(int)
    n_ig = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            zip(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort(kind='mergesort')[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]
            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            n_ig[l] += gt_difficult_l.sum()
            score[l].extend(pred_score_l)
            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
            # VOC evaluation follows integer typed bounding boxes.
            # pred_bbox_l = pred_bbox_l.copy()
            # pred_bbox_l[:, 2:] += 1
            # gt_bbox_l = gt_bbox_l.copy()
            # gt_bbox_l[:, 2:] += 1
            iou = bbox_iou(pred_bbox_l, gt_bbox_l, gt_difficult_l)
            gt_iou = iou[:, ~gt_difficult_l]
            ig_iou = iou[:, gt_difficult_l]
            if ig_iou.shape[1] > 0:
                ig_index = ig_iou.max(axis=1) > iou_thresh
            else:
                ig_index = np.zeros([pred_bbox_l.shape[0], 1], dtype=np.bool)

            # set -1 if there is no matching ground truth
            # selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for dt_i in range(iou.shape[0]):
                if gt_iou.shape[1] > 0:
                    gt_idx = gt_iou[dt_i, :].argmax()
                    if gt_iou[dt_i, gt_idx] >= iou_thresh:
                        match[l].append(1)
                        gt_iou[:, gt_idx] = 0
                        continue
                if ig_index[dt_i]:
                    match[l].append(-1)
                else:
                    match[l].append(0)
    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    miss_rate = [None] * n_fg_class
    false_positive_per_image = [None] * n_fg_class
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort(kind='mergesort')[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
        # ig = np.cumsum(match_l == -1)
        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        false_positive_per_image[l] = (fp / N)
        # If n_pos[l] is 0, rec[l] is None.
        # print(f'一共{N}张图片，{n_pos[l]}个目标，{n_ig[l]}个忽略区域，预测对{tp[-1] if tp != [] else 0}个目标，错{fp[-1] if fp != [] else 0}个目标，忽略{ig[-1] if ig != [] else 0}')
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            miss_rate[l] = 1 - rec[l]
    return prec, rec, miss_rate, false_positive_per_image


def calc_detection_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]
            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_detection_mr(miss_rate, FPPI, FPPI_sample):
    n_fg_class = len(miss_rate)
    MR = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if miss_rate[l] is None or FPPI[l] is None:
            MR[l] = np.nan
            continue
            # 11 point metric
        MR[l] = 0
        for t in np.float_power(10, np.arange(start=FPPI_sample[0], stop=FPPI_sample[1] + 0.2, step=0.25)):
            if np.sum(FPPI[l] <= t) == 0:
                p = 1
            else:
                p = np.min(np.nan_to_num(miss_rate[l], nan=1)[FPPI[l] <= t])
            MR[l] += np.log(p) / 9
    return np.exp(MR)
