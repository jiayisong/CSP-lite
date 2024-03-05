import torch.nn

from models.init_weight import *
from config import args
import torch.nn.functional as F


class head(nn.Module):
    def __init__(self, n_cls, input_channel):
        super(head, self).__init__()
        self.n_cls = n_cls
        size_channel = 1
        self.heat_map_conv = nn.Sequential(
            nn.Conv2d(input_channel, n_cls, kernel_size=1, stride=1, padding=0, bias=False),
        )
        nn.init.normal_(self.heat_map_conv[0].weight, std=0.01)
        self.size_conv = nn.Sequential(
            nn.Conv2d(input_channel, size_channel, kernel_size=1, stride=1, padding=0, bias=False),
        )
        nn.init.normal_(self.size_conv[0].weight, std=0.01)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(input_channel, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )
        nn.init.normal_(self.offset_conv[0].weight, std=0.01)

    def forward(self, x):
        heat = self.heat_map_conv(x)
        heat = torch.sigmoid(heat - 4.6)
        size = self.size_conv(x)
        offset = self.offset_conv(x)
        return heat, size, offset
