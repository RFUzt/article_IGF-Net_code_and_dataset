"""
SegMAN - closer to the original paper implementation
Reference: "SegMAN: Omni-Scale Context Modeling with State Space Models and
           Local Attention for Semantic Segmentation" (arXiv 2024)
Official repository: https://github.com/1e12Leon/SegMAN
This implementation approximates S6 behavior using PyTorch without relying on the mamba_ssm library,
while retaining key architectural designs from the original: shifted window, relative position bias,
4-direction cross-scan SSM, and an omni-scale decoder with channel attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchinfo import summary
from utils.registry import register_model


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__();
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training: return x
        keep = 1 - self.p
        s = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = torch.rand(s, device=x.device).floor_().div_(keep)
        return x * r

def window_partition(x, win):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
    x = x.view(B, H // win, win, W // win, win, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win * win, C)
    return windows


def window_reverse(windows, win, H, W):
    B = int(windows.shape[0] / ((H // win) * (W // win)))
    C = windows.shape[-1]
    x = windows.view(B, H // win, W // win, win, win, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x.permute(0, 3, 1, 2)


class RelativePositionBias(nn.Module):
    def __init__(self, win, heads):
        super().__init__()
        self.win = win
        self.table = nn.Parameter(
            torch.zeros((2 * win - 1) * (2 * win - 1), heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords_h = torch.arange(win)
        coords_w = torch.arange(win)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.flatten(1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += win - 1
        rel[:, :, 1] += win - 1
        rel[:, :, 0] *= 2 * win - 1
        idx = rel.sum(-1)
        self.register_buffer("idx", idx)

    def forward(self):
        return self.table[self.idx].permute(2, 0, 1).unsqueeze(0)


class LocalWindowAttention(nn.Module):
    def __init__(self, dim, win=7, heads=4, shift=False, qkv_bias=True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.win = win
        self.heads = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.shift = shift
        self.shift_size = win // 2 if shift else 0

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rpb = RelativePositionBias(win, heads)

    def _attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        slices_h = (slice(0, -self.win),
                    slice(-self.win, -self.shift_size),
                    slice(-self.shift_size, None))
        slices_w = (slice(0, -self.win),
                    slice(-self.win, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[:, sh, sw, :] = cnt;
                cnt += 1
        mask_windows = img_mask.view(1, H // self.win, self.win,
                                     W // self.win, self.win, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, self.win * self.win)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.).masked_fill(attn_mask == 0, 0.)
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        win = self.win

        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

        w = window_partition(x, win)
        w = self.norm(w)

        BnW, N, _ = w.shape
        qkv = self.qkv(w).reshape(BnW, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb()

        mask = self._attn_mask(Hp, Wp, x.device)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(BnW, self.heads, N, N)

        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).contiguous().reshape(BnW, N, C)
        out = self.proj(out)

        x = window_reverse(out, win, Hp, Wp)


        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        x = x[:, :, :H, :W]
        return x


class SelectiveScan2D(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_inner = int(dim * expand)
        d_inner = self.d_inner

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)

        # DWConv on x branch
        self.dwconv = nn.Conv2d(d_inner, d_inner, d_conv,
                                padding=d_conv // 2, groups=d_inner, bias=True)
        self.act = nn.SiLU()

        self.d_state = d_state
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_inner, 1)
        self.register_buffer("A_log", torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def _ssm_scan(self, u):

        N, L, D = u.shape
        d_state = self.d_state

        dt = F.softplus(self.dt_proj(u))
        B = self.B_proj(u)
        C = self.C_proj(u)

        A = -torch.exp(self.A_log.float())

        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(N, D, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t, :].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * self.D
        return y

    def _cross_scan(self, x2d):

        B, H, W, D = x2d.shape

        u_rf = x2d.reshape(B * H, W, D)
        y_rf = self._ssm_scan(u_rf).view(B, H, W, D)

        u_rb = torch.flip(u_rf, [1])
        y_rb = torch.flip(self._ssm_scan(u_rb), [1]).view(B, H, W, D)

        u_cf = x2d.permute(0, 2, 1, 3).contiguous().reshape(B * W, H, D)
        y_cf = self._ssm_scan(u_cf).view(B, W, H, D).permute(0, 2, 1, 3).contiguous()
        # col backward
        u_cb = torch.flip(u_cf, [1])
        y_cb = torch.flip(self._ssm_scan(u_cb), [1]).view(B, W, H, D).permute(0, 2, 1, 3).contiguous()

        return y_rf + y_rb + y_cf + y_cb

    def forward(self, x):
        B, C, H, W = x.shape

        seq = x.flatten(2).transpose(1, 2)
        seq = self.norm(seq)

        xz = self.in_proj(seq)
        x_, z = xz.chunk(2, dim=-1)

        x_2d = x_.view(B, H, W, -1).permute(0, 3, 1, 2)
        x_2d = self.act(self.dwconv(x_2d))
        x_2d = x_2d.permute(0, 2, 3, 1)

        y = self._cross_scan(x_2d)
        y = self.out_norm(y)

        z_2d = self.act(z.view(B, H, W, -1))
        y = y * z_2d

        y = self.out_proj(y.reshape(B, H * W, -1))
        return y.transpose(1, 2).view(B, C, H, W)


class SegMANBlock(nn.Module):
    def __init__(self, dim, win=7, heads=4, shift=False, drop_path=0.,
                 mlp_ratio=4., d_state=16):
        super().__init__()
        self.lwa = LocalWindowAttention(dim, win=win, heads=heads, shift=shift)
        self.ssm = SelectiveScan2D(dim, d_state=d_state)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)
        self.dp3 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp1(self.lwa(x))
        x = x + self.dp2(self.ssm(x))

        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = seq + self.dp3(self.ffn(self.norm_ffn(seq)))
        x = seq.transpose(1, 2).view(B, C, H, W)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, dim // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return self.proj(x)


class PatchMerging(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.op(x)


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * w


class OmniScaleContext(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1, bias=False)

        self.local_dw = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim, bias=False)
        self.local_pw = nn.Conv2d(out_dim, out_dim, 1, bias=False)

        self.fuse = nn.Sequential(
            ConvBNAct(out_dim * 4, out_dim, 1, 1, 0),
            ConvBNAct(out_dim, out_dim, 3, 1, 1),
            ChannelAttention(out_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x0 = self.proj(x)

        p_g = F.adaptive_avg_pool2d(x0, 1)
        p_g = F.interpolate(p_g, (H, W), mode='bilinear', align_corners=False)

        p_h = F.adaptive_avg_pool2d(x0, (max(1, H // 2), max(1, W // 2)))
        p_h = F.interpolate(p_h, (H, W), mode='bilinear', align_corners=False)

        local = self.local_pw(self.local_dw(x0))

        y = torch.cat([x0, local, p_g, p_h], dim=1)
        return self.fuse(y)


class SegMANv2(nn.Module):

    def __init__(
            self,
            in_channels=3,
            num_classes=19,
            base_dim=64,
            depths=(2, 2, 6, 2),
            heads=(1, 2, 4, 8),
            win=7,
            d_state=16,
            mlp_ratio=4.,
            drop_path_rate=0.1,
            dec_dim=256,
            apply_sigmoid=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.apply_sigmoid = apply_sigmoid

        dims = [base_dim * 2 ** i for i in range(4)]

        self.patch_embed = PatchEmbed(in_channels, dims[0])

        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]
        idx = 0

        def make_stage(dim, n, nh):
            nonlocal idx
            blocks = []
            for i in range(n):
                blocks.append(SegMANBlock(
                    dim, win=win, heads=nh,
                    shift=(i % 2 == 1),
                    drop_path=dpr[idx],
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                ))
                idx += 1
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(dims[0], depths[0], heads[0])
        self.down1 = PatchMerging(dims[0], dims[1])

        self.stage2 = make_stage(dims[1], depths[1], heads[1])
        self.down2 = PatchMerging(dims[1], dims[2])

        self.stage3 = make_stage(dims[2], depths[2], heads[2])
        self.down3 = PatchMerging(dims[2], dims[3])

        self.stage4 = make_stage(dims[3], depths[3], heads[3])

        self.ctx4 = OmniScaleContext(dims[3], dec_dim)

        self.lat3 = ConvBNAct(dims[2], dec_dim, 1, 1, 0, act=False)
        self.lat2 = ConvBNAct(dims[1], dec_dim, 1, 1, 0, act=False)
        self.lat1 = ConvBNAct(dims[0], dec_dim, 1, 1, 0, act=False)

        self.fuse3 = ConvBNAct(dec_dim, dec_dim, 3, 1, 1)
        self.fuse2 = ConvBNAct(dec_dim, dec_dim, 3, 1, 1)
        self.fuse1 = ConvBNAct(dec_dim, dec_dim, 3, 1, 1)

        self.head = nn.Sequential(
            ConvBNAct(dec_dim, dec_dim // 2, 3, 1, 1),
            nn.Conv2d(dec_dim // 2, num_classes, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight);
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out_size = x.shape[-2:]

        c1 = self.stage1(self.patch_embed(x))
        c2 = self.stage2(self.down1(c1))
        c3 = self.stage3(self.down2(c2))
        c4 = self.stage4(self.down3(c3))

        p4 = self.ctx4(c4)

        def up(feat, lateral, fuse):
            return fuse(F.interpolate(feat, size=lateral.shape[-2:],
                                      mode='bilinear', align_corners=False) + lateral)

        p3 = up(p4, self.lat3(c3), self.fuse3)
        p2 = up(p3, self.lat2(c2), self.fuse2)
        p1 = up(p2, self.lat1(c1), self.fuse1)

        y = F.interpolate(p1, size=out_size, mode='bilinear', align_corners=False)
        out = self.head(y)

        if self.num_classes == 1 and self.apply_sigmoid:
            out = torch.sigmoid(out)
        return out


@register_model("segmanv2")
def segman_default(in_channels=3, num_classes=19, **kw):
    return SegMANv2(in_channels=in_channels, num_classes=num_classes,
                  base_dim=64, depths=(2, 2, 6, 2), heads=(1, 2, 4, 8), **kw)


if __name__ == "__main__":
    m = SegMANv2(in_channels=5, num_classes=19)
    summary(m, input_size=(1, 5, 256, 256), device="cpu")