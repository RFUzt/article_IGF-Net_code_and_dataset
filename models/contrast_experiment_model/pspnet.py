# models/pspnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.registry import register_model


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=None):
        super().__init__()
        mid = out_ch // self.expansion

        self.conv1 = ConvBNReLU(in_ch, mid, k=1, s=1, p=0)
        self.conv2 = ConvBNReLU(mid, mid, k=3, s=stride, p=dilation, d=dilation)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class ResNetBackboneOS8(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()
        self.inplanes = base_ch

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(outplanes=base_ch * 4, blocks=3, stride=1, dilation=1)
        self.layer2 = self._make_layer(outplanes=base_ch * 8, blocks=4, stride=2, dilation=1)
        self.layer3 = self._make_layer(outplanes=base_ch * 16, blocks=6, stride=1, dilation=2)
        self.layer4 = self._make_layer(outplanes=base_ch * 32, blocks=3, stride=1, dilation=4)

    def _make_layer(self, outplanes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, outplanes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, outplanes, stride=1, dilation=dilation, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_ch, out_ch=512, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        assert in_ch % 4 == 0, "PSP PPM usually uses in_ch divisible by 4"
        inter_ch = in_ch // 4

        self.branches = nn.ModuleList()
        for ps in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    ConvBNReLU(in_ch, inter_ch, k=1, s=1, p=0),
                )
            )

        self.fusion = nn.Sequential(
            ConvBNReLU(in_ch + len(pool_sizes) * inter_ch, out_ch, k=3, s=1, p=1),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for b in self.branches:
            y = b(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        x = torch.cat(feats, dim=1)
        return self.fusion(x)


class PSPNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        apply_sigmoid=True,
        base_ch=64,
        ppm_out_ch=512,
        use_auxiliary=True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.use_auxiliary = bool(use_auxiliary)

        self.backbone = ResNetBackboneOS8(in_channels=in_channels, base_ch=base_ch)

        c4_ch = base_ch * 16
        c5_ch = base_ch * 32

        self.ppm = PyramidPoolingModule(in_ch=c5_ch, out_ch=ppm_out_ch, pool_sizes=(1, 2, 3, 6))

        self.main_head = nn.Sequential(
            ConvBNReLU(ppm_out_ch, ppm_out_ch // 2, k=3, s=1, p=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(ppm_out_ch // 2, self.num_classes, kernel_size=1),
        )

        if self.use_auxiliary:
            self.aux_head = nn.Sequential(
                ConvBNReLU(c4_ch, ppm_out_ch // 4, k=3, s=1, p=1),
                nn.Dropout2d(0.1),
                nn.Conv2d(ppm_out_ch // 4, self.num_classes, kernel_size=1),
            )
        else:
            self.aux_head = None

    def forward(self, x, return_aux=False):

        in_size = x.shape[-2:]

        _, c4, c5 = self.backbone(x)

        psp = self.ppm(c5)
        main_out = self.main_head(psp)
        main_out = F.interpolate(main_out, size=in_size, mode="bilinear", align_corners=False)

        aux_out = None
        if self.use_auxiliary and self.aux_head is not None:
            aux_out = self.aux_head(c4)
            aux_out = F.interpolate(aux_out, size=in_size, mode="bilinear", align_corners=False)

        if self.num_classes == 1 and self.apply_sigmoid:
            main_out = torch.sigmoid(main_out)
            if aux_out is not None:
                aux_out = torch.sigmoid(aux_out)

        if return_aux:
            return main_out, aux_out

        return main_out


@register_model("pspnet")
def build_pspnet(in_channels=3, num_classes=1, apply_sigmoid=True, **kwargs):
    return PSPNet(
        in_channels=in_channels,
        num_classes=num_classes,
        apply_sigmoid=apply_sigmoid,
        **kwargs
    )


if __name__ == "__main__":
    model = PSPNet(in_channels=5, num_classes=1, apply_sigmoid=True)
    summary(model, input_size=(1, 5, 256, 256), device="cpu")
