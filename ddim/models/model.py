import math
import torch
import torch.nn as nn

class TimeMLP(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,)
        return self.net(t[:, None])  # (B, time_dim)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.silu = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.silu(self.gn1(self.conv1(x)))
        scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.silu(self.gn2(self.conv2(h)))
        return (self.skip(x) + h) / math.sqrt(2.0)


class UNetMNIST(nn.Module):
    def __init__(
        self, 
        base_channels: int = 64, 
        time_dim: int = 256, 
        img_channels: int = 3,
        num_classes: int = 10
    ):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim)
        self.class_emb = nn.Embedding(num_classes, time_dim)

        self.in_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        # Down path
        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        # Up path
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_label: torch.Tensor | None = None) -> torch.Tensor:
        # expect t already scaled to [0,1]
        t_emb = self.time_mlp(t)
        if class_label is not None:
            class_emb = self.class_emb(class_label)
            t_emb = t_emb + class_emb

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)           # 28×28
        d2_in = self.downsample1(d1)         # 14×14
        d2 = self.down2(d2_in, t_emb)
        d3_in = self.downsample2(d2)         # 7×7

        h = self.bottleneck(d3_in, t_emb)

        h = self.upsample2(h)                # 14×14
        h = torch.cat([h, d2], 1)
        h = self.up2(h, t_emb)

        h = self.upsample1(h)                # 28×28
        h = torch.cat([h, d1], 1)
        h = self.up1(h, t_emb)

        return self.out_conv(h)