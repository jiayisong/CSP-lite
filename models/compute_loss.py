import math
import torch.nn as nn
import torch
from config import args


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.heat_pos_weight = args.loss_weight[0]
        self.size_weight = args.loss_weight[1]
        self.offset_weight = args.loss_weight[2]


    def forward(self, outs, targets):
        heatmap_pos, heatmap_neg, size, mask = targets
        pre_heatmap, pre_size, pre_offset = outs
        batchsize, _, height, width = pre_size.size()
        heat_loss, heat_pos_loss, heat_neg_loss = self.focal_loss(pre_heatmap, heatmap_pos, heatmap_neg)
        reg_loss, size_error, offset_error = self.L1_loss(pre_size, pre_offset, size, mask)
        loss = heat_loss + reg_loss
        return [loss, heat_pos_loss.item(), heat_neg_loss.item(), size_error.item(), offset_error.item(), 0]

    def focal_loss(self, pred, heatmap_pos, heatmap_neg):
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        pos_num = heatmap_pos.sum()
        neg_num = heatmap_neg.sum()
        pos_num = torch.maximum(pos_num, torch.tensor(1).to(pos_num))
        pos_loss = heatmap_pos * torch.log(pred) * torch.pow(1 - pred, 2)
        neg_loss = heatmap_neg * torch.log(1 - pred) * torch.pow(pred, 2)
        pos_loss = - pos_loss.sum()
        neg_loss = - neg_loss.sum()
        if args.NNL:
            loss = (pos_loss + neg_loss) * self.heat_pos_weight
        else:
            loss = (pos_loss + neg_loss) * self.heat_pos_weight / pos_num
        return loss, pos_loss / pos_num, neg_loss / neg_num


    def L1_loss(self, pre_size, pre_offset, gt, mask):
        # pre_l, pre_t, pre_r, pre_b = torch.split(pre, split_size_or_sections=1, dim=1)
        gt_w, gt_h, gt_offset_x, gt_offset_y = torch.split(gt, split_size_or_sections=1, dim=1)
        # gt_h = torch.log(gt_t + gt_b)

        gt_size = torch.cat([gt_w, gt_h], dim=1)
        gt_offset = torch.cat((gt_offset_x, gt_offset_y), dim=1)
        size_mask = mask
        offset_mask = mask
        size_num = size_mask.sum()
        offset_num = offset_mask.sum()
        if pre_size.size(1) > 1:
            size_error = torch.log(gt_size) - pre_size
            size_num = torch.maximum(2 * size_num, torch.tensor(1).to(size_num))
        else:
            size_error = torch.log(gt_h) - pre_size
            size_num = torch.maximum(size_num, torch.tensor(1).to(size_num))
        offset_error = pre_offset - gt_offset
        offset_num = torch.maximum(2 * offset_num, torch.tensor(1).to(offset_num))
        l1_loss = torch.abs(size_error) * size_mask
        l1_offset_loss = torch.abs(offset_error) * offset_mask
        if args.NNL:
            l1_loss = l1_loss.sum()
            l1_offset_loss = l1_offset_loss.sum()
        else:
            l1_loss = l1_loss.sum() / size_num
            l1_offset_loss = l1_offset_loss.sum() / offset_num
        loss = l1_loss * self.size_weight + l1_offset_loss * self.offset_weight
        with torch.no_grad():
            size_error2 = (torch.abs(size_error) * size_mask).sum() / size_num
            offset_error2 = (torch.abs(offset_error / gt_size) * offset_mask).sum() / offset_num
        return loss, size_error2, offset_error2

