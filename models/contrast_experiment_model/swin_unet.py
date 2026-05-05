import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.registry import register_model


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


def pad_to_window_size(x, window_size):

    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = x.shape[1], x.shape[2]
    return x, (Hp, Wp), (pad_h, pad_w)


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)

        size = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))

        coords = torch.stack(torch.meshgrid(
            torch.arange(self.window_size),
            torch.arange(self.window_size),
            indexing="ij"
        ))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self):
        index = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[index]
        N = self.window_size * self.window_size
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        return bias


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = self.dim // self.num_heads
        assert head_dim * self.num_heads == self.dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpb = RelativePositionBias(self.window_size, self.num_heads)

    def forward(self, x, attn_mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.rpb()[None, :, :, :]

        if attn_mask is not None:

            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask[None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = input_resolution
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(self.dim, int(self.dim * self.mlp_ratio), drop=drop)

        self.register_buffer("_attn_mask_hw", torch.zeros(2, dtype=torch.long), persistent=False)
        self.register_buffer("_attn_mask", None, persistent=False)

    def _build_attn_mask(self, H, W, device):

        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, H, W):

        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x, (Hp, Wp), (pad_h, pad_w) = pad_to_window_size(x, self.window_size)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        if self.shift_size > 0:
            hw = torch.tensor([Hp, Wp], device=x.device, dtype=torch.long)
            if (self._attn_mask is None) or (self._attn_mask_hw[0] != hw[0]) or (self._attn_mask_hw[1] != hw[1]):
                self._attn_mask = self._build_attn_mask(Hp, Wp, x.device)
                self._attn_mask_hw = hw
            attn_mask = self._attn_mask
        else:
            attn_mask = None

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)

    def forward(self, x, H, W):

        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = x.shape[1]
            W = x.shape[2]

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchExpand(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.expand = nn.Linear(self.dim, 2 * self.dim, bias=False)
        self.norm = nn.LayerNorm(self.dim // 2)

    def forward(self, x, H, W):

        B, L, C = x.shape
        assert L == H * W
        x = self.expand(x)
        x = x.view(B, H, W, 2 * C)

        x = x.view(B, H, W, 2, C).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, H * 2, W, C)

        x = x.view(B, H * 2, W, 2, C // 2).permute(0, 1, 2, 3, 4).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)

        H2, W2, C2 = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, H2 * W2, C2)
        x = self.norm(x)
        return x, H2, W2


class FinalPatchExpandX4(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.expand = nn.Linear(self.dim, 16 * self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x, H, W):

        B, L, C = x.shape
        assert L == H * W
        x = self.expand(x)
        x = x.view(B, H, W, 16 * self.out_dim)

        x = x.view(B, H, W, 4, 4, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 4, W * 4, self.out_dim)

        x = x.view(B, -1, self.out_dim)
        x = self.norm(x)
        return x, H * 4, W * 4


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rates=None,
        downsample=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.window_size = int(window_size)

        if drop_path_rates is None:
            drop_path_rates = [0.0] * self.depth

        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            shift_size = 0 if (i % 2 == 0) else (self.window_size // 2)
            self.blocks.append(
                SwinTransformerBlock(
                    dim=self.dim,
                    input_resolution=None,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                )
            )

        self.downsample = downsample(self.dim) if downsample is not None else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rates=None,
    ):
        super().__init__()
        self.up = PatchExpand(dim)
        self.concat_linear = nn.Linear(dim // 2 + dim // 2, dim // 2, bias=False)

        self.layer = BasicLayer(
            dim=dim // 2,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path_rates=drop_path_rates,
            downsample=None
        )

    def forward(self, x, H, W, skip, Hs, Ws):
        x, H, W = self.up(x, H, W)

        if H != Hs or W != Ws:

            B, L, C = x.shape
            xf = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            xf = F.interpolate(xf, size=(Hs, Ws), mode="bilinear", align_corners=False)
            x = xf.permute(0, 2, 3, 1).contiguous().view(B, Hs * Ws, C)
            H, W = Hs, Ws

        x = torch.cat([x, skip], dim=-1)
        x = self.concat_linear(x)

        x, H, W = self.layer(x, H, W)
        return x, H, W


class SwinUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        img_size=128,
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        apply_sigmoid=True
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.apply_sigmoid = bool(apply_sigmoid)

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size=patch_size)

        total_depth = sum(depths) + sum(depths[:-1])
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        dp_i = 0

        self.enc1 = BasicLayer(
            dim=embed_dim,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[0]],
            downsample=PatchMerging
        )
        dp_i += depths[0]

        self.enc2 = BasicLayer(
            dim=embed_dim * 2,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[1]],
            downsample=PatchMerging
        )
        dp_i += depths[1]

        self.enc3 = BasicLayer(
            dim=embed_dim * 4,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[2]],
            downsample=PatchMerging
        )
        dp_i += depths[2]

        self.enc4 = BasicLayer(
            dim=embed_dim * 8,
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[3]],
            downsample=None
        )
        dp_i += depths[3]

        self.dec3 = DecoderLayer(
            dim=embed_dim * 8,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[2]],
        )
        dp_i += depths[2]

        self.dec2 = DecoderLayer(
            dim=embed_dim * 4,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[1]],
        )
        dp_i += depths[1]

        self.dec1 = DecoderLayer(
            dim=embed_dim * 2,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rates=dpr[dp_i: dp_i + depths[0]],
        )
        dp_i += depths[0]

        self.final_up = FinalPatchExpandX4(embed_dim, embed_dim)

        self.head = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

    def forward(self, x):

        B, _, H0, W0 = x.shape

        x, H, W = self.patch_embed(x)

        x1, H1, W1 = x, H, W
        x, H, W = self.enc1(x, H, W)
        x2, H2, W2 = x, H, W

        x, H, W = self.enc2(x, H, W)
        x3, H3, W3 = x, H, W

        x, H, W = self.enc3(x, H, W)
        x4, H4, W4 = x, H, W

        x, H, W = self.enc4(x, H, W)

        x, H, W = self.dec3(x, H, W, skip=x3, Hs=H3, Ws=W3)
        x, H, W = self.dec2(x, H, W, skip=x2, Hs=H2, Ws=W2)
        x, H, W = self.dec1(x, H, W, skip=x1, Hs=H1, Ws=W1)

        x, H, W = self.final_up(x, H, W)

        xf = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if xf.shape[-2:] != (H0, W0):
            xf = F.interpolate(xf, size=(H0, W0), mode="bilinear", align_corners=False)

        out = self.head(xf)

        if self.num_classes == 1 and self.apply_sigmoid:
            out = torch.sigmoid(out)

        return out


@register_model("swin_unet")
def build_swin_unet(in_channels=3, num_classes=1, apply_sigmoid=True, **kwargs):
    return SwinUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        apply_sigmoid=apply_sigmoid,
        **kwargs
    )


if __name__ == "__main__":
    model = SwinUNet(
        in_channels=14,
        num_classes=1,
        window_size=4,
        drop_path_rate=0.2,
    )
    summary(model, input_size=(1, 14, 128, 128), device="cpu")
