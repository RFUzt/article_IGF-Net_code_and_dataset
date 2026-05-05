# models/unet.py

import torch
import torch.nn as nn
from torchinfo import summary
from utils.registry import register_model


def conv_block(in_ch, out_ch):

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

class UNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, base_ch=64, apply_sigmoid=True):
        super().__init__()
        self.apply_sigmoid = bool(apply_sigmoid)
        self.num_classes = int(num_classes)

        self.enc1 = conv_block(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch*8, base_ch*16)

        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = conv_block(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)

        if self.num_classes == 1 and self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out

@register_model("unet")
def build_unet(in_channels=3, num_classes=1, apply_sigmoid=True, **kwargs):
    return UNet(in_channels=in_channels, num_classes=num_classes, apply_sigmoid=apply_sigmoid, **kwargs)


if __name__ == "__main__":
    model = UNet(in_channels=14, num_classes=1)
    summary(model, input_size=(1, 14, 128, 128), device="cpu")
