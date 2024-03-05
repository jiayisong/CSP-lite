import torch.nn as nn
from models.resnet_CSP import backbone as backbone
from models.neck_CSP import neck as neck
from models.head_CSP import head as head
import models.decode as dc


class my_model(nn.Module):
    def __init__(self, n_cls):
        super(my_model, self).__init__()
        self.backbone = backbone()
        self.neck = neck(input_channel=self.backbone.output_channel)
        self.head = head(input_channel=self.neck.output_channel, n_cls=n_cls)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def merge_bn_conv(self):
        self.back_bone.merge_bn_conv()
        self.detection.merge_bn_conv()


class my_model_with_decode(nn.Module):
    def __init__(self, n_cls):
        super(my_model_with_decode, self).__init__()
        self.my_module = my_model(n_cls)

    def forward(self, x):
        pre = self.my_module(x)
        dec = dc._decode(pre)
        return dec
