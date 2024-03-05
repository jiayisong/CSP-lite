from torchvision.ops import batched_nms
import torch


def _topk(scores, K):
    batch, _, height, width = scores.size()
    hw = height * width
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds // hw).int()
    topk_inds = topk_inds - hw * topk_clses
    topk_inds = topk_inds
    ys = topk_inds // width
    ys = ys.int()
    topk_ys = ys.float()
    topk_xs = (topk_inds - ys * width).float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs, batch, height, width


def _decode(pre, K=1000):
    heat, size, offset = pre
    scores, inds, clses, ys, xs, batch, height, width = _topk(heat, K=K)
    size = size.view([batch, size.size(1), -1])
    inds = inds.unsqueeze(1)
    inds = inds.expand([batch, size.size(1), K])
    size = size.gather(dim=2, index=inds)
    size = size.permute([0, 2, 1])
    size = size.exp()
    if size.size(2) == 1:
        size = torch.cat((0.41 * size, size), dim=2)
    inds = inds.expand([batch, 2, K])
    offset = offset.view([batch, 2, -1])
    offset = offset.gather(dim=2, index=inds)
    offset = offset.permute([0, 2, 1])
    center = torch.stack((xs, ys), dim=2) + offset
    tl = center - size * 0.5
    br = center + size * 0.5
    bboxes = torch.cat((tl, br), dim=2)
    return bboxes, scores, clses


def predict(dec):
    bboxes, scores, clses = dec
    nms = 0.5
    scores_threshold = 0.01
    bbox = torch.split(bboxes, 1, dim=0)
    score = torch.split(scores, 1, dim=0)
    cls = torch.split(clses, 1, dim=0)
    predictions = []
    for b, s, c in zip(bbox, score, cls):
        b = b.squeeze(0)
        s = s.squeeze(0)
        c = c.squeeze(0)
        ind = s > scores_threshold
        s = s[ind]
        b = b[ind, :]
        c = c[ind]
        if nms and len(s) > 0:
            ind = batched_nms(b, s, c, nms)
            b = torch.index_select(b, 0, ind)
            c = torch.index_select(c, 0, ind)
            s = torch.index_select(s, 0, ind)
        b = b.cpu().numpy()
        s = s.cpu().numpy()
        c = c.cpu().numpy()
        prediction = {'boxes': b, 'labels': c,
                      'scores': s}
        predictions.append(prediction)
    return predictions


def decode(pre):
    dec = _decode(pre)
    return predict(dec)
