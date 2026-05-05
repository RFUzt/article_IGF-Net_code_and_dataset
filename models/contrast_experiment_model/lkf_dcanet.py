# A Lightweight Network for Water Body Segmentation in Agricultural Remote Sensing Using Learnable Kalman Filters and Attention Mechanisms
# https://github.com/Estrellading/-A-Lightweight-Network-for-Water-Body-Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.registry import register_model

try:
    from torchvision.ops import DeformConv2d
    HAS_DEFORM = True
except:
    HAS_DEFORM = False


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        mid = max(ch // r, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CADCN(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        if HAS_DEFORM:
            self.offset = nn.Conv2d(in_ch, 18, 3, padding=1)
            self.deform = DeformConv2d(in_ch, out_ch, 3, padding=1, bias=False)
        else:
            self.deform = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.ca = ChannelAttention(out_ch)

        self.refine = nn.Sequential(
            ConvBNAct(out_ch, out_ch),
            ConvBNAct(out_ch, out_ch)
        )

        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        if HAS_DEFORM:
            off = self.offset(x)
            x = self.deform(x, off)
        else:
            x = self.deform(x)

        x = self.act(self.bn(x))
        x = self.ca(x)
        x = self.refine(x)
        x = self.down(x)
        return x


class CATM(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.dw(x) + self.pw(x)
        y = self.act(self.bn(y))
        return x + y


class LearnableKalmanFilter(nn.Module):
    """
    Multi-stage LKF
    """
    def __init__(self, ch):
        super().__init__()
        self.K = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )
        self.state = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, z):
        B, C, H, W = z.shape
        x_hat = self.state.expand(B, C, H, W)
        K = self.K(z)
        return x_hat + K * (z - x_hat)


class LKF_DCANet(nn.Module):

    def __init__(self, in_ch=3, num_classes=1, base=32, apply_sigmoid=True):
        super().__init__()
        self.num_classes = num_classes
        self.apply_sigmoid = apply_sigmoid

        self.stem = ConvBNAct(in_ch, base)

        self.e1 = CADCN(base, base * 2)
        self.e2 = CADCN(base * 2, base * 4)
        self.e3 = CADCN(base * 4, base * 8)

        self.bottleneck = nn.Sequential(
            ConvBNAct(base * 8, base * 8),
            CATM(base * 8),
            ConvBNAct(base * 8, base * 8)
        )

        self.up2 = nn.Sequential(
            CATM(base * 8),
            nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        )
        self.lkf2 = LearnableKalmanFilter(base * 4)

        self.up1 = nn.Sequential(
            CATM(base * 4),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        )
        self.lkf1 = LearnableKalmanFilter(base * 2)

        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.lkf0 = LearnableKalmanFilter(base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]

        x0 = self.stem(x)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        b = self.bottleneck(x3)

        d2 = self.lkf2(self.up2(b))
        d1 = self.lkf1(self.up1(d2))
        d0 = self.lkf0(self.up0(d1))

        out = self.head(d0)
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)

        if self.num_classes == 1 and self.apply_sigmoid:
            out = torch.sigmoid(out)

        return out


@register_model("lkf_dcanet")
def build_lkf_dcanet(in_channels=3, num_classes=1, **kwargs):
    return LKF_DCANet(in_ch=in_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = LKF_DCANet(in_ch=14, num_classes=1)
    summary(model, input_size=(1, 14, 128, 128), device="cpu")
