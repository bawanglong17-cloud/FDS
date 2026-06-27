from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def chebyshev_t(order: int, x: torch.Tensor) -> torch.Tensor:
    if order == 0:
        return torch.ones_like(x)
    if order == 1:
        return x
    t0 = torch.ones_like(x)
    t1 = x
    for _ in range(1, order):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return t1


def l2_normalize_kernel(k: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    k = k - k.mean()
    return k / torch.sqrt(torch.sum(k * k) + eps)


def make_chebyshev_split_kernels(kernel_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be odd.")

    x = torch.linspace(-1.0, 1.0, steps=kernel_size)

    # 低阶结构模板：平滑响应
    t0 = chebyshev_t(0, x)
    low = torch.outer(t0, t0)
    low = low / low.sum().clamp_min(1e-8)

    # 高阶细节模板：振荡响应
    t2 = chebyshev_t(2, x)
    high = torch.outer(t2, t2)
    high = l2_normalize_kernel(high)

    return low.float(), high.float()


class ChebyshevSplit(nn.Module):
    """
    切比雪夫初始化的结构/细节分流层。
    输入 : [B, C, H, W]
    输出 : xs结构特征, xd细节特征，均为 [B, C, H, W]
    """

    def __init__(self, channels: int, kernel_size: int = 3, trainable: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.split = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.reset_parameters()
        self.split.weight.requires_grad_(trainable)

    def reset_parameters(self) -> None:
        low, high = make_chebyshev_split_kernels(self.kernel_size)
        with torch.no_grad():
            self.split.weight.zero_()
            for c in range(self.channels):
                self.split.weight[2 * c, 0].copy_(low)
                self.split.weight[2 * c + 1, 0].copy_(high)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        f = self.split(x).view(b, c, 2, h, w)
        xs = f[:, :, 0, :, :]
        xd = f[:, :, 1, :, :]
        return xs, xd


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetailCNNBranch(nn.Module):
    """细节分支：两层 3x3 Conv-BN-ReLU。"""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            ConvBNReLU(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupedDenseWavKAN2d(nn.Module):
    """
    分组 Dense WAV-KAN。

    完整 Dense WAV-KAN 会让每个输出通道看全部 Cin*3*3 patch，
    显存压力非常大。这里改为分组版本：
    每个输出通道只看本组通道的 3x3 patch，但仍有组内通道交互，
    表达力强于 depthwise，显存低于完整 Dense。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        groups: int = 8,
        spatial_chunk: int = 512,
        eps: float = 1e-4,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd.")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}.")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by groups={groups}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.in_per_group = in_channels // groups
        self.out_per_group = out_channels // groups
        self.patch_dim_group = self.in_per_group * kernel_size * kernel_size
        self.spatial_chunk = spatial_chunk
        self.eps = eps

        self.translation = nn.Parameter(torch.zeros(groups, self.out_per_group, self.patch_dim_group))
        self.log_scale = nn.Parameter(torch.zeros(groups, self.out_per_group, self.patch_dim_group))
        self.weight = nn.Parameter(torch.empty(groups, self.out_per_group, self.patch_dim_group))
        self.bias = nn.Parameter(torch.zeros(groups, self.out_per_group))

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def _forward_chunk(self, patches_g: torch.Tensor, g: int) -> torch.Tensor:
        """
        patches_g: [B, Lc, Dg]
        return:    [B, Lc, Og]
        """
        scale = F.softplus(self.log_scale[g]) + self.eps  # [Og, Dg]

        z = (
            patches_g[:, None, :, :]
            - self.translation[g][None, :, None, :]
        ) / scale[None, :, None, :]

        # Mexican-hat/Ricker-like wavelet
        psi = (1.0 - z.pow(2)) * torch.exp(-0.5 * z.pow(2))
        y = (psi * self.weight[g][None, :, None, :]).sum(dim=-1)
        y = y + self.bias[g][None, :, None]
        y = y.permute(0, 2, 1).contiguous()  # [B, Lc, Og]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}.")

        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )  # [B, Cin*K*K, L]

        l = h * w
        patches = patches.view(
            b,
            self.groups,
            self.in_per_group * self.kernel_size * self.kernel_size,
            l,
        )
        patches = patches.permute(0, 1, 3, 2).contiguous()
        # [B, G, L, Dg]

        out_groups = []
        for g in range(self.groups):
            patches_g = patches[:, g]  # [B, L, Dg]
            chunk_outputs = []
            for start in range(0, l, self.spatial_chunk):
                end = min(start + self.spatial_chunk, l)
                chunk_outputs.append(self._forward_chunk(patches_g[:, start:end], g))

            yg = torch.cat(chunk_outputs, dim=1)  # [B, L, Og]
            out_groups.append(yg)

        y = torch.cat(out_groups, dim=-1)  # [B, L, Cout]
        y = y.permute(0, 2, 1).contiguous().view(b, self.out_channels, h, w)

        y = self.norm(y)
        y = self.act(y)
        return y


class BottleneckGroupedWavKAN2d(nn.Module):
    """
    低显存结构分支：
        1x1 降维: C -> C/r
        Grouped Dense WAV-KAN
        1x1 升维: C/r -> C
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        groups: int = 8,
        kernel_size: int = 3,
        spatial_chunk: int = 512,
        min_hidden: int = 32,
    ):
        super().__init__()

        hidden = max(channels // reduction, min_hidden)
        hidden = int(math.ceil(hidden / groups) * groups)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.wavkan = GroupedDenseWavKAN2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=kernel_size,
            groups=groups,
            spatial_chunk=spatial_chunk,
        )

        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z = self.wavkan(z)
        z = self.expand(z)
        return z


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ECAAttention(nn.Module):
    """ECA: Efficient Channel Attention."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                    # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)     # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class CBAMAttention(nn.Module):
    """CBAM: Channel attention + spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.channel_sigmoid = nn.Sigmoid()

        self.spatial = nn.Sequential(
            nn.Conv2d(
                2,
                1,
                kernel_size=spatial_kernel,
                padding=spatial_kernel // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * self.channel_sigmoid(avg_out + max_out)

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_map, max_map], dim=1))
        return x * spatial_att


def build_fusion_attention(attention_type: str, channels: int, reduction: int = 16) -> nn.Module:
    attention_type = str(attention_type).lower()

    if attention_type in ["ours", "channel", "channel_attention", "ca"]:
        return ChannelAttention(channels, reduction=reduction)

    if attention_type == "eca":
        return ECAAttention(channels)

    if attention_type == "cbam":
        return CBAMAttention(channels, reduction=reduction)

    raise ValueError(
        f"Unknown fusion_attention: {attention_type}. "
        "Choose from: ours, eca, cbam."
    )


class FDSBlockLite(nn.Module):
    """
    低显存版 FDSBlock：
    切比雪夫分流 -> CNN细节分支 / Bottleneck Grouped WAV-KAN结构分支 -> 1x1融合 -> 通道注意力 -> 残差。
    """

    def __init__(
        self,
        channels: int,
        split_kernel_size: int = 3,
        kan_kernel_size: int = 3,
        kan_reduction: int = 4,
        kan_groups: int = 8,
        kan_spatial_chunk: int = 512,
        attention_reduction: int = 16,
        split_trainable: bool = True,
        fusion_attention: str = "ours",
    ):
        super().__init__()

        self.split = ChebyshevSplit(
            channels=channels,
            kernel_size=split_kernel_size,
            trainable=split_trainable,
        )

        self.detail_branch = DetailCNNBranch(channels)

        self.structure_branch = BottleneckGroupedWavKAN2d(
            channels=channels,
            reduction=kan_reduction,
            groups=kan_groups,
            kernel_size=kan_kernel_size,
            spatial_chunk=kan_spatial_chunk,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.att = build_fusion_attention(fusion_attention, channels, reduction=attention_reduction)
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs, xd = self.split(x)

        f_kan = self.structure_branch(xs)
        f_cnn = self.detail_branch(xd)

        fused = self.fuse(torch.cat([f_kan, f_cnn], dim=1))
        fused = self.att(fused)

        out = self.out_act(self.out_norm(fused + x))
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3),
            ConvBNReLU(out_ch, out_ch, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Baseline U-Net，不含 FDS。"""

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        base_ch: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5 + c4, c4, bilinear)
        self.up2 = Up(c4 + c3, c3, bilinear)
        self.up3 = Up(c3 + c2, c2, bilinear)
        self.up4 = Up(c2 + c1, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class FDSUNetLite(nn.Module):
    """FDS-UNet：在 U-Net 瓶颈层加入低显存 FDS。"""

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        base_ch: int = 64,
        bilinear: bool = True,
        kan_reduction: int = 4,
        kan_groups: int = 8,
        kan_spatial_chunk: int = 512,
        fusion_attention: str = "ours",
    ):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.fds = FDSBlockLite(
            channels=c5,
            kan_reduction=kan_reduction,
            kan_groups=kan_groups,
            kan_spatial_chunk=kan_spatial_chunk,
            fusion_attention=fusion_attention,
        )

        self.up1 = Up(c5 + c4, c4, bilinear)
        self.up2 = Up(c4 + c3, c3, bilinear)
        self.up3 = Up(c3 + c2, c2, bilinear)
        self.up4 = Up(c2 + c1, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.fds(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


class FDSForSwinBottleneckLite(nn.Module):
    """
    给 Swin-UNet 瓶颈 token 使用的 FDS 适配器。
    输入:
    - [B, C, H, W] 直接处理
    - [B, L, C] 需要传 h, w，输出仍为 [B, L, C]
    """

    def __init__(self, channels: int, **fds_kwargs):
        super().__init__()
        self.fds = FDSBlockLite(channels, **fds_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> torch.Tensor:
        if x.dim() == 4:
            return self.fds(x)

        if x.dim() != 3:
            raise ValueError("x must be [B,C,H,W] or [B,L,C].")

        b, l, c = x.shape
        if h is None or w is None:
            side = int(math.sqrt(l))
            if side * side != l:
                raise ValueError("For token input [B,L,C], please provide h and w.")
            h = w = side

        feat = x.transpose(1, 2).contiguous().view(b, c, h, w)
        feat = self.fds(feat)
        out = feat.flatten(2).transpose(1, 2).contiguous()
        return out


def build_model(
    model_name: str,
    in_channels: int = 3,
    n_classes: int = 1,
    base_ch: int = 64,
    kan_reduction: int = 4,
    kan_groups: int = 8,
    kan_spatial_chunk: int = 512,
    fusion_attention: str = "ours",
) -> nn.Module:
    name = model_name.lower()
    if name in ["unet", "baseline"]:
        return UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            base_ch=base_ch,
        )

    if name in ["fds_unet", "fds_unet_lite", "fdsunet", "fdsunet_lite"]:
        return FDSUNetLite(
            in_channels=in_channels,
            n_classes=n_classes,
            base_ch=base_ch,
            kan_reduction=kan_reduction,
            kan_groups=kan_groups,
            kan_spatial_chunk=kan_spatial_chunk,
            fusion_attention=fusion_attention,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
