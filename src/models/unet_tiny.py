# Lightweight U-Net Tiny (~0.1M params) with legacy arg aliases (in_ch/out_ch)
# - Keeps compatibility with eval scripts calling UNetTiny(in_ch=3, out_ch=1)

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv3x3(in_ch, out_ch),
            _conv3x3(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch)
            self.is_deconv = False
        else:
            # deconv 입력 채널은 in_ch//2 로 가정 (표준 U-Net 패턴)
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
            self.is_deconv = True

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.is_deconv:
            x = self.up(x)
        else:
            x = self.up(x)

        # pad if needed
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetTiny(nn.Module):
    """
    Very small U-Net:
      enc: 16-32-64  /  dec: 64-32-16  (≈0.11M params)
    Arguments (with legacy aliases):
      - in_channels (alias: in_ch)   default 3
      - out_channels(alias: out_ch)  default 1
      - bilinear: use bilinear upsample instead of transposed conv
      - base_ch: base channel size (default 16). Keep at 16 to stay ~0.1M params.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        *,
        in_ch: int | None = None,
        out_ch: int | None = None,
        bilinear: bool = True,
        base_ch: int = 16,
        **kwargs,
    ):
        super().__init__()
        if in_ch is not None:
            in_channels = in_ch
        if out_ch is not None:
            out_channels = out_ch

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_ch = base_ch
        self.bilinear = bilinear

        c1 = base_ch          # 16
        c2 = base_ch * 2      # 32
        c3 = base_ch * 4      # 64

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)

        # Bottleneck
        self.bot = DoubleConv(c3, c3)

        # Decoder
        # bilinear: Up(in_ch = cat channels), where upsample doesn't change channels
        # deconv  : Up(in_ch = cat channels), but internal ConvTranspose2d uses in_ch//2
        self.up1 = Up(c3 + c2, c2, bilinear=bilinear)  # concat bot with enc2
        self.up2 = Up(c2 + c1, c1, bilinear=bilinear)  # concat with enc1

        # Head
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # [B, c1, H, W]
        x2 = self.down1(x1)    # [B, c2, H/2, W/2]
        x3 = self.down2(x2)    # [B, c3, H/4, W/4]

        xb = self.bot(x3)      # [B, c3, H/4, W/4]

        u1 = self.up1(xb, x2)  # in_ch = c3+c2 -> out = c2
        u2 = self.up2(u1, x1)  # in_ch = c2+c1 -> out = c1

        logits = self.outc(u2) # [B, out_ch, H, W]
        return logits


if __name__ == "__main__":
    net = UNetTiny(in_ch=3, out_ch=1, base_ch=16)
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print("out:", tuple(y.shape), "params(M):", sum(p.numel() for p in net.parameters())/1e6)
