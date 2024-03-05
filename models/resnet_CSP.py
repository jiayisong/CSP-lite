from torchvision.models.resnet import resnet18
import torch.nn as nn
import torch


class backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(backbone, self).__init__()
        self.resnet = resnet18(pretrained=pretrained, replace_stride_with_dilation=[False, False, False])
        self.output_channel = [64, 128, 256, 512]

    def forward(self, x):
        # print((x.max(),x.min()),x.mean())
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)

        C3 = self.resnet.layer2(x)
        C4 = self.resnet.layer3(C3)
        C5 = self.resnet.layer4(C4)
        return C3, C4, C5
