# models/deeplabv3plus.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models import resnet50
from utils.registry import register_model



class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[0],
                      dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[1],
                      dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[2],
                      dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        bg = self.global_pool(x)
        bg = F.interpolate(bg, size=(h, w),
                           mode="bilinear",
                           align_corners=False)

        out = torch.cat([b1, b2, b3, b4, bg], dim=1)
        return self.project(out)



class DeepLabV3Plus(nn.Module):

    def __init__(self,
                 in_channels=3,
                 num_classes=1,
                 output_stride=16,
                 apply_sigmoid=True,
                 pretrained=False):

        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.num_classes = num_classes

        backbone = resnet50(weights=None)

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            aspp_rates = (6, 12, 18)
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_rates = (12, 24, 36)
        else:
            raise ValueError("output_stride must be 8 or 16")

        backbone = resnet50(
            weights=None,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.aspp = ASPP(2048, 256, rates=aspp_rates)


        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x = self.layer0(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)

        x = self.aspp(high)
        x = F.interpolate(x,
                          size=low.shape[2:],
                          mode="bilinear",
                          align_corners=False)

        low = self.low_conv(low)

        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)

        x = F.interpolate(x,
                          size=(h, w),
                          mode="bilinear",
                          align_corners=False)

        if self.num_classes == 1 and self.apply_sigmoid:
            x = torch.sigmoid(x)

        return x


@register_model("deeplabv3plus")
def build_deeplabv3plus(in_channels=3,
                                 num_classes=1,
                                 apply_sigmoid=True,
                                 **kwargs):
    return DeepLabV3Plus(
        in_channels=in_channels,
        num_classes=num_classes,
        apply_sigmoid=apply_sigmoid,
        **kwargs
    )


if __name__ == "__main__":
    model = DeepLabV3Plus(
        in_channels=14,
        num_classes=1,
        output_stride=16
    )

    summary(model, input_size=(1, 14, 128, 128))
