# models/fcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.registry import register_model


def conv_block(in_ch, out_ch):

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

class FCN32s(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, base_ch=64, apply_sigmoid=True):
        super().__init__()
        self.apply_sigmoid = bool(apply_sigmoid)
        self.num_classes = int(num_classes)

        self.layer1 = conv_block(in_channels, base_ch);      self.pool1 = nn.MaxPool2d(2)
        self.layer2 = conv_block(base_ch, base_ch*2);        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = conv_block(base_ch*2, base_ch*4);      self.pool3 = nn.MaxPool2d(2)
        self.layer4 = conv_block(base_ch*4, base_ch*8);      self.pool4 = nn.MaxPool2d(2)
        self.layer5 = conv_block(base_ch*8, base_ch*8);      self.pool5 = nn.MaxPool2d(2)

        self.classifier = nn.Conv2d(base_ch*8, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = self.pool5(self.layer5(x))
        x = self.classifier(x)
        out = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if self.num_classes == 1 and self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out


@register_model("fcn32s")
def build_fcn(in_channels=3, num_classes=1, apply_sigmoid=True, **kwargs):
    return FCN32s(in_channels=in_channels, num_classes=num_classes, apply_sigmoid=apply_sigmoid, **kwargs)


if __name__ == "__main__":
    model = FCN32s(in_channels=14, num_classes=1)
    summary(model, input_size=(14, 128, 128), device="cpu")
