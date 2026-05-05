# models/IGFNet.py
"""
IGF-Net implementation:
    - Dual encoder: RGB (pretrained ResNet50) + multispectral (11/2 channels trained from scratch) to extract complementary features.
    - IGFM: intelligent gated fusion based on (x⊙y, |x-y|), with channel prior, adaptive temperature, and residual fidelity.
    - ASPP: Atrous Spatial Pyramid Pooling to aggregate multi-scale context from high-level features.
    - Multi-level decoder: fuses low-level (/4), high-level (/8), and original encoder features, gradually upsampling to restore resolution.
    - Supports in_channels=14 / 5 (multispectral) / num_classes; optional Sigmoid for binary classification.
    - Registered via @register_model("igf-net").
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils.registry import register_model


def load_timm_checkpoint(model, ckpt_path: str, strict: bool = False, verbose: bool = True):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model" in ckpt):
        state_dict = ckpt.get("state_dict", ckpt.get("model"))
    else:
        state_dict = ckpt

    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)

    if verbose:
        print(f"[load_timm_checkpoint] loaded from: {ckpt_path}")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")

    return missing, unexpected


class IntelligentGateFusion(nn.Module):

    def __init__(
        self,
        in_channels: int,
        reduction: int = 8,
        t_min: float = 0.6,
        t_max: float = 1.8,
        dilation: int = 1,
        use_channel_prior: bool = True,
        use_refine: bool = True
    ):
        super().__init__()
        C = in_channels
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.use_channel_prior = use_channel_prior
        self.use_refine = use_refine

        self.head = nn.Sequential(
            nn.Conv2d(2*C, 2*C, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(2*C),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*C, 2*C, kernel_size=3, padding=dilation, dilation=dilation,
                      groups=2*C, bias=False),
            nn.BatchNorm2d(2*C)
        )

        self.tau_head = nn.Sequential(
            nn.Conv2d(2*C, C, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=dilation, dilation=dilation,
                      groups=C, bias=False),
            nn.BatchNorm2d(C)
        )

        if use_channel_prior:
            hidden = max(1, C // reduction)
            self.prior_mlp = nn.Sequential(
                nn.Linear(2*C, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2*C, bias=True)
            )

            nn.init.zeros_(self.prior_mlp[-1].bias)

        if use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(C, C, kernel_size=1, bias=False),
                nn.BatchNorm2d(C)
            )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        B, C, H, W = x.shape
        assert y.shape == x.shape, "x and y must have the same shape"

        xy   = x * y
        diff = (x - y).abs()
        s = torch.cat([xy, diff], dim=1)

        logits = self.head(s).view(B, 2, C, H, W)

        if self.use_channel_prior:
            g_xy   = F.adaptive_avg_pool2d(xy, 1).view(B, C)
            g_diff = F.adaptive_avg_pool2d(diff, 1).view(B, C)
            g = torch.cat([g_xy, g_diff], dim=1)
            prior = self.prior_mlp(g).view(B, 2, C, 1, 1)
            logits = logits + prior

        tau_logits = self.tau_head(s)
        tau = torch.sigmoid(tau_logits)
        tau = self.t_min + (self.t_max - self.t_min) * tau

        logits_scaled = logits / tau.unsqueeze(1)
        weights = torch.softmax(logits_scaled, dim=1)
        w_x = weights[:, 0, ...]
        w_y = weights[:, 1, ...]

        fused = w_x * x + w_y * y

        if self.use_refine:
            fused = self.refine(fused)
        out = fused + self.gamma * (x + y) * 0.5
        return out


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels=256, rates=(1, 2, 4)):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        result = torch.cat([x1, x2, x3, x4, x5], dim=1)

        return result


class IGFNet(nn.Module):
    def __init__(self, in_channels=14, num_classes=1, apply_sigmoid=True):
        super(IGFNet, self).__init__()

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        if self.in_channels == 14:
            rgb_indices = [6, 10, 11]
            mult_channels = 11
        elif self.in_channels == 5:
            rgb_indices = [0, 1, 2]
            mult_channels = 2
        else:
            rgb_indices = [0, 1, 2]
            mult_channels = self.in_channels - 3
            print(
                f"Warning: Unexpected input channels {self.in_channels}, using first 3 channels as RGB, {mult_channels} channels for multispectral")

        self.rgb_indices = rgb_indices
        self.mult_channels = mult_channels

        mult_base_encoder = timm.create_model(model_name="resnet50", pretrained=False,
                                              features_only=True, in_chans=self.mult_channels, output_stride=8)

        self.mult_encoder1 = nn.Sequential(*list(mult_base_encoder.children())[0:3])
        self.mult_encoder2 = nn.Sequential(*list(mult_base_encoder.children())[3:5])

        rgb_ckpt = r"pre_trained_model\timm_resnet50\pytorch_model.bin"
        """
        Here is the actual path where the user downloads the pre-trained model. It is recommended to use an absolute path.
        """
        rgb_base_encoder = timm.create_model(
            model_name="resnet50",
            pretrained=False,
            features_only=True,
            in_chans=3,
            output_stride=8
        )
        load_timm_checkpoint(rgb_base_encoder, rgb_ckpt, strict=False, verbose=True)

        self.rgb_encoder1 = nn.Sequential(*list(rgb_base_encoder.children())[0:3])
        self.rgb_encoder2 = nn.Sequential(*list(rgb_base_encoder.children())[3:5])

        self.assemble_layer = IntelligentGateFusion(in_channels=256)

        self.rgb_encoder3 = nn.Sequential(*list(rgb_base_encoder.children())[5:6])
        self.rgb_encoder4 = nn.Sequential(*list(rgb_base_encoder.children())[6:7])
        self.rgb_encoder5 = nn.Sequential(*list(rgb_base_encoder.children())[7:8])

        self.aspp = ASPP(in_channels=2048)

        self.decoder_high_layer = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.Up1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)

        self.decoder_lower_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.Up2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)


        self.decoder5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.Up3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)

        self.decoder6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.decoder7 = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )

        self.apply_sigmoid = bool(apply_sigmoid)

    def forward(self, x):
        mult_indices = [i for i in range(self.in_channels) if i not in self.rgb_indices]

        rgb = x[:, self.rgb_indices, :, :]
        mult = x[:, mult_indices, :, :]

        mult1 = self.mult_encoder1(mult)
        mult2 = self.mult_encoder2(mult1)

        rgb_1 = self.rgb_encoder1(rgb)
        rgb_2 = self.rgb_encoder2(rgb_1)

        low_level = self.assemble_layer(rgb_2, mult2)
        low_level_decoder = self.decoder_lower_layer(low_level)

        rgb_3 = self.rgb_encoder3(low_level)
        rgb_4 = self.rgb_encoder4(rgb_3)
        rgb_5 = self.rgb_encoder5(rgb_4)

        high_level = self.aspp(rgb_5)
        high_level_decoder = self.decoder_high_layer(high_level)

        high_level_decoder = self.Up1(high_level_decoder)

        decoder_assemble1 = torch.cat((low_level_decoder, high_level_decoder), dim=1)

        decoder3 = self.decoder3(decoder_assemble1)

        decoder_assemble2 = torch.cat((mult2, rgb_2, decoder3), dim=1)

        decoder4 = self.decoder4(decoder_assemble2)
        decoder4 = self.Up2(decoder4)

        decoder5 = self.decoder5(decoder4)
        decoder5 = self.Up3(decoder5)

        decoder6 = self.decoder6(decoder5)

        output = self.decoder7(decoder6)

        if self.num_classes == 1 and self.apply_sigmoid:
            output = torch.sigmoid(output)

        return output


@register_model("igf-net")
def build_IGFNet(in_channels=5, num_classes=1, apply_sigmoid=True, **kwargs):
    return IGFNet(in_channels=in_channels, num_classes=num_classes, apply_sigmoid=apply_sigmoid, **kwargs)


if __name__ == "__main__":
    model = IGFNet(in_channels=14, num_classes=1)
    summary(model, input_size=(14, 256, 256), device="cpu")