from config import args
import torch
from models.init_weight import *



class neck(nn.Module):
    def __init__(self, input_channel):
        super(neck, self).__init__()
        self.FPN = FPN(in_channel=input_channel)
        self.output_channel = self.FPN.output_channel

    def forward(self, x):
        return self.FPN(x)


class FPN(nn.Module):
    def __init__(self, in_channel):
        super(FPN, self).__init__()
        self.deconv1 = nn.Sequential(my_upsample(in_channel[-1], 4),
                                     )
        self.deconv2 = nn.Sequential(my_upsample(in_channel[-2], 2),
                                     )
        self.deconv3 = nn.Sequential(  # my_downsample(in_channel[-3], 2),
        )

        d16 = in_channel[-1] // 4 + in_channel[-2] + in_channel[-3] * 4
        d8 = in_channel[-1] // 16 + in_channel[-2] // 4 + in_channel[-3]
        g = 2
        if args.neck == 'res':
            self.conv = nn.Sequential(
                nn.Conv2d(d8, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                *[BasicBlock(256) for i in range(args.neck_depth)]
            )
        elif args.neck == 'mulbn':
            self.conv = nn.Sequential(
                nn.Conv2d(d8, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                *[MBNM(256, 3, g) for i in range(args.neck_depth)]
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(d8, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        norm_initialize_weights(self.conv[0])
        self.output_channel = 256  # in_channel[-1] // 16 + in_channel[-2] // 4 + in_channel[-3]

    def forward(self, x):
        C3, C4, C5 = x
        C5 = self.deconv1(C5)
        C4 = self.deconv2(C4)
        C3 = self.deconv3(C3)
        x = torch.cat((C3, C4, C5), dim=1)
        x = self.conv(x)
        return x


class my_upsample(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(my_upsample, self).__init__()
        self.scale_factor_sqaure = scale_factor ** 2
        assert (in_channel % self.scale_factor_sqaure) == 0
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c // self.scale_factor_sqaure, self.scale_factor, self.scale_factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c // self.scale_factor_sqaure, h * self.scale_factor, w * self.scale_factor)
        return x


class my_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(my_conv, self).__init__()
        assert out_channels % groups == 0
        self.groups = groups
        a = 0.1
        per_groups = out_channels // groups
        self.conv = nn.Conv2d(in_channels, per_groups, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        bn_weight = torch.zeros_like(self.bn.weight)
        bn_bias = torch.zeros_like(self.bn.bias)
        w = 1
        for i in range(groups):
            bn_weight[i * per_groups:(i + 1) * per_groups] = w
            # nn.init.constant_(bn.bias, val=random.uniform(-(groups-1)*a, (groups-1)*a))
            if i < 2:
                bn_bias[i * per_groups:(i + 1) * per_groups] = 0
            else:
                bn_bias[i * per_groups: (i + 1) * per_groups] = (i + 2) // 4 * a * ((-1) ** (i // 2))
            w = -w
            # self.bn.append(bn)
        self.bn.weight.data = bn_weight
        self.bn.bias.data = bn_bias
        norm_initialize_weights(self.conv)

    def forward(self, x):
        x = self.conv(x)
        g = self.groups
        if g > 1:
            b, c, h, w = x.size()
            x = x.view(b, 1, c, h, w)
            x = x.expand(b, g, c, h, w)
            x = x.reshape(b, c * g, h, w)
        return self.bn(x)


class MBNM(nn.Module):
    def __init__(self, in_channels, kernel_size, groups):
        super(MBNM, self).__init__()
        self.conv = nn.Sequential(
            my_conv(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2, groups=groups),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            my_conv(in_channels, in_channels, kernel_size, stride=1, padding=kernel_size // 2, groups=groups),
            # nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        # norm_initialize_weights(self.conv[0])
        # norm_initialize_weights(self.conv[3])

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class BasicBlock(nn.Module):
    def __init__(self, inplanes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        norm_initialize_weights(self.conv1)
        norm_initialize_weights(self.conv2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
