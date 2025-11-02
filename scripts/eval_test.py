#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm


# =========================================================
# Dataset: CVC-ClinicDB (images/, masks/)
# =========================================================
class CVCTestDataset(Dataset):
    def __init__(self, root="data/CVC-ClinicDB", img_size=(256, 256), norm: str = "none"):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"
        if not (self.img_dir.exists() and self.msk_dir.exists()):
            raise FileNotFoundError(
                f"Expected folders:\n  {self.img_dir}\n  {self.msk_dir}\n"
                "Each with *.png having the same stem."
            )

        imgs = sorted([p for p in self.img_dir.glob("*.png")])
        msks = sorted([p for p in self.msk_dir.glob("*.png")])
        img_stems = {p.stem for p in imgs}
        msk_stems = {p.stem for p in msks}
        common = sorted(list(img_stems & msk_stems))
        if not common:
            raise FileNotFoundError(f"No matched *.png stems under {self.img_dir} & {self.msk_dir}")

        self.pairs: List[Tuple[Path, Path]] = [
            (self.img_dir / f"{s}.png", self.msk_dir / f"{s}.png") for s in common
        ]
        self.img_size = img_size
        self.norm = norm

        if norm == "imagenet":
            self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        img = img.resize(self.img_size, Image.BILINEAR)
        msk = msk.resize(self.img_size, Image.NEAREST)

        x = TF.to_tensor(img)  # (3,H,W) in [0,1]
        if self.mean is not None:
            x = (x - self.mean) / self.std
        y = torch.from_numpy((np.array(msk) > 127).astype(np.float32))[None]  # (1,H,W)
        return x, y, ip.stem


def make_test_loader_cvc(batch_size=1, img_size=(256, 256), norm="none"):
    return DataLoader(
        CVCTestDataset(img_size=img_size, norm=norm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


# =========================================================
# Models
# =========================================================
def _import_unet_tiny():
    from src.models.unet_tiny import UNetTiny  # must exist in your repo
    return UNetTiny


def _conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv3x3(in_ch, out_ch),
            _conv3x3(out_ch, out_ch),
        )
    def forward(self, x): return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))


class _UpDeconv(nn.Module):
    """Legacy up block with ConvTranspose2d then DoubleConv on concat."""
    def __init__(self, up_in: int, up_out: int, dec_in_after_concat: int, dec_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
        self.conv = _DoubleConv(dec_in_after_concat, dec_out)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy or dx:
            x = nn.functional.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LegacyRawUNet(nn.Module):
    """
    EXACT legacy layout to match ckpt keys WITHOUT remap:
      enc1.block.*, enc2.block.*, bott.{0,1,3,4}.*, dec2.*, dec1.*, out.*, up1, up2
    Channels from ckpt evidence:
      c1=16, c2=32, bott=64
      up2: 64->32 deconv, dec2 in=(32(↑) + 32)=64 -> out 32
      up1: 32->16 deconv, dec1 in=(16(↑) + 16)=32 -> out 16
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        c1, c2, c3 = 16, 32, 64
        # encoders
        self.enc1 = _DoubleConv(in_ch, c1)          # keys: enc1.block.*
        self.enc2 = nn.Sequential(                   # enc2.block.*
            nn.MaxPool2d(2),
            _DoubleConv(c1, c2)
        )
        # bottleneck: bott.0 (32->64), bott.1 BN64, bott.3 (64->64), bott.4 BN64
        self.bott = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1, bias=True), nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=True), nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        # dec2/up2: up2 64->32, dec2 (32+32=64 -> 32)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)                 # keys: up2.weight/bias
        self.dec2 = _DoubleConv(c2 + c2, c2)                         # dec2.block.*
        # dec1/up1: up1 32->16, dec1 (16+16=32 -> 16)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)                 # keys: up1.weight/bias
        self.dec1 = _DoubleConv(c1 + c1, c1)                         # dec1.block.*
        # head
        self.out = nn.Conv2d(c1, out_ch, 1)                          # out.weight/bias

    def forward(self, x):
        x1 = self.enc1(x)               # (B,16,H,W)
        x2 = self.enc2(x1)              # (B,32,H/2,W/2)
        xb = self.bott(x2)              # (B,64,H/2,W/2)  (note: enc2 uses pool, bott keeps H/2)
        u2 = self.up2(xb)               # (B,32,H,W)
        u2 = torch.cat([u2, x2], 1)     # (B,64,H,W)
        d2 = self.dec2(u2)              # (B,32,H,W)
        u1 = self.up1(d2)               # (B,16,H*2,W*2)  but x1 is (H,W), so pad handles if off-by-odd
        dy = x1.size(-2) - u1.size(-2)
        dx = x1.size(-1) - u1.size(-1)
        if dy or dx:
            u1 = nn.functional.pad(u1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        u1 = torch.cat([u1, x1], 1)     # (B,32,H,W)
        d1 = self.dec1(u1)              # (B,16,H,W)
        y = self.out(d1)                # (B,1,H,W)
        return y


# =========================================================
# Legacy → Current state_dict key remap
# =========================================================
def remap_legacy_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k

        # direct legacy path stays untouched (for legacy_raw)
        # for remap path, transform names:
        if nk.startswith("enc1."):
            nk = nk.replace("enc1.", "inc.")
        elif nk.startswith("enc2."):
            nk = nk.replace("enc2.", "down1.conv.")
        elif nk.startswith("dec2."):
            nk = nk.replace("dec2.", "up1.conv.")
        elif nk.startswith("dec1."):
            nk = nk.replace("dec1.", "up2.conv.")
        elif nk.startswith("out."):
            nk = nk.replace("out.", "outc.")

        # bott.* -> down2.conv.*
        if k.startswith("bott.0.") or k.startswith("bott.1."):
            nk = re.sub(r"^bott\.(0|1)\.", "down2.conv.", nk, count=1)
        elif k.startswith("bott.3.") or k.startswith("bott.4."):
            nk = re.sub(r"^bott\.(3|4)\.", "down2.conv.", nk, count=1)

        # block -> net
        nk = nk.replace(".block.", ".net.")

        # index expansion to .net.{0,1}.{0,1}
        nk = nk.replace(".0.", ".net.0.0.")
        nk = nk.replace(".1.", ".net.0.1.")
        nk = nk.replace(".3.", ".net.1.0.")
        nk = nk.replace(".4.", ".net.1.1.")

        out[nk] = v
    return out


# =========================================================
# Weight fitting utilities (relaxed load)
# =========================================================
def _fit_1d_param(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    out_c = dst_shape[0]
    if src.ndim != 1:
        return src
    if src.shape[0] == out_c:
        return src
    if src.shape[0] > out_c:
        return src[:out_c].clone()
    pad = out_c - src.shape[0]
    return torch.cat([src, torch.zeros(pad, dtype(src))], dim=0)


def dtype(x: torch.Tensor): return x.dtype


def _fit_running_stat(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    if dst_shape == torch.Size([]):
        return torch.tensor(0, dtype=src.dtype)
    return _fit_1d_param(src, dst_shape)


def _fit_conv_weight(src: torch.Tensor, dst_shape: torch.Size) -> torch.Tensor:
    # Conv2d weight: [out_c, in_c, kH, kW]
    if src.ndim != 4 or len(dst_shape) != 4:
        return src
    so, si, kh, kw = src.shape
    do, di, kH, kW = dst_shape

    w = src
    # kernel adjust (center crop / pad)
    if (kh != kH) or (kw != kW):
        pad_h = max(0, kH - kh)
        pad_w = max(0, kW - kw)
        if pad_h or pad_w:
            w = nn.functional.pad(w, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
        if w.shape[2] > kH:
            t = (w.shape[2]-kH)//2
            w = w[:, :, t:t+kH, :]
        if w.shape[3] > kW:
            t = (w.shape[3]-kW)//2
            w = w[:, :, :, t:t+kW]

    # out channels
    if w.shape[0] > do:
        w = w[:do]
    elif w.shape[0] < do:
        pad = torch.zeros((do - w.shape[0], w.shape[1], w.shape[2], w.shape[3]), dtype=w.dtype)
        w = torch.cat([w, pad], dim=0)

    # in channels
    if w.shape[1] > di:
        w = w[:, :di]
    elif w.shape[1] < di:
        pad = torch.zeros((w.shape[0], di - w.shape[1], w.shape[2], w.shape[3]), dtype=w.dtype)
        w = torch.cat([w, pad], dim=1)

    return w


def _fit_any(src: torch.Tensor, dst: torch.Tensor, key: str) -> Optional[torch.Tensor]:
    if src.shape == dst.shape:
        return src
    if src.ndim == 4 and dst.ndim == 4:
        return _fit_conv_weight(src, dst.shape)
    if src.ndim == 1 and dst.ndim == 1:
        return _fit_1d_param(src, dst.shape)
    if dst.ndim in (0, 1) and src.ndim in (0, 1) and (
        key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked")
    ):
        return _fit_running_stat(src, dst.shape)
    return None


def relaxed_load(model: nn.Module, sd: Dict[str, torch.Tensor], resize_convs: bool = True):
    msd = model.state_dict()
    filled = resized = skipped = 0
    new_sd = {}
    for k, dst in msd.items():
        if k in sd:
            src = sd[k]
            if src.shape == dst.shape:
                new_sd[k] = src; filled += 1
            else:
                fitted = _fit_any(src, dst, k) if resize_convs else None
                if fitted is not None and fitted.shape == dst.shape:
                    new_sd[k] = fitted; resized += 1
                else:
                    skipped += 1
        else:
            skipped += 1
    model.load_state_dict(new_sd, strict=False)
    return filled, resized, skipped


# =========================================================
# Helpers
# =========================================================
def detect_legacy_names(sd: Dict[str, torch.Tensor]) -> bool:
    prefixes = ("enc1.", "enc2.", "bott.", "dec1.", "dec2.", "out.")
    return any(any(k.startswith(p) for p in prefixes) for k in sd.keys())


def dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def run_eval(args):
    # device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # ckpt
    tmp = torch.load(args.ckpt, map_location="cpu")
    sd_raw: Dict[str, torch.Tensor] = tmp.get("model", tmp)

    # choose load mode
    load_mode = args.load_mode  # 'relaxed' or 'legacy_raw'
    use_legacy_names = detect_legacy_names(sd_raw)

    if load_mode == "legacy_raw":
        # ------------- strict legacy path (no remap) -------------
        print("[info] load_mode=legacy_raw (strict) -> using LegacyRawUNet and raw keys")
        model = LegacyRawUNet(in_ch=3, out_ch=1).to(device).eval()
        missing, unexpected = model.load_state_dict(sd_raw, strict=False)
        if missing or unexpected:
            print(f"[warn] strict raw-load mismatch -> missing={len(missing)}, unexpected={len(unexpected)}")
        # data
        od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True); (od/"viz").mkdir(exist_ok=True)
        loader = make_test_loader_cvc(batch_size=1, img_size=(args.img_size, args.img_size), norm=args.norm)
        return _eval_loop(args, device, model, loader, od)

    # ------------- default: remap + relaxed -------------
    if use_legacy_names:
        sd_map = remap_legacy_keys(sd_raw)
        down2_cnt = sum(k.startswith("down2.conv.") for k in sd_map.keys())
        print(f"[info] remap applied -> down2 keys: {down2_cnt}")
        if down2_cnt < 8:
            print("[warn] down2 remap seems low; falling back to relaxed anyway.")

        UNetTiny = _import_unet_tiny()
        # infer base_ch
        base_ch_guess = None
        for probe in ("inc.net.0.0.weight", "inc.net.0.weight", "inc.net.1.weight"):
            if probe in sd_map and sd_map[probe].ndim >= 1:
                base_ch_guess = int(sd_map[probe].shape[0]); break
        if base_ch_guess is None:
            for probe in ("enc1.block.0.weight", "enc1.0.weight"):
                if probe in sd_raw and sd_raw[probe].ndim >= 1:
                    base_ch_guess = int(sd_raw[probe].shape[0]); break
        if base_ch_guess is None:
            base_ch_guess = args.base_ch or 16

        model = UNetTiny(in_channels=3, out_channels=1, base_ch=base_ch_guess, bilinear=True).to(device).eval()
        print("[info] Legacy/deconv-like ckpt detected -> using UNetTiny(bilinear=True) with relaxed load")
        filled, resized, skipped = relaxed_load(model, sd_map, resize_convs=True)
        print(f"[info] relaxed load: filled={filled}, resized={resized}, skipped={skipped}")

    else:
        # if it wasn't legacy at all, just try plain load to UNetTiny
        UNetTiny = _import_unet_tiny()
        model = UNetTiny(in_channels=3, out_channels=1, base_ch=args.base_ch or 16, bilinear=True).to(device).eval()
        model.load_state_dict(sd_raw, strict=False)

    # data
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True); (od/"viz").mkdir(exist_ok=True)
    loader = make_test_loader_cvc(batch_size=1, img_size=(args.img_size, args.img_size), norm=args.norm)
    return _eval_loop(args, device, model, loader, od)


def _eval_loop(args, device, model, loader, od: Path):
    # thresholds
    ths = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)] if args.th_sweep else [float(args.th)]
    sums = {th: {"dice": 0.0, "iou": 0.0, "n": 0} for th in ths}

    saved = 0
    for x, y, stem in tqdm(loader, desc="eval", ncols=80):
        x = x.to(device); y = y.to(device)
        prob = torch.sigmoid(model(x))

        for th in ths:
            pred = (prob >= th).float()
            d, j = dice_iou(pred, y)
            sums[th]["dice"] += d
            sums[th]["iou"] += j
            sums[th]["n"] += 1

        if saved < args.vis_n:
            img_vis = x.detach().cpu()
            if args.norm == "imagenet":
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                img_vis = (img_vis * std + mean).clamp(0, 1)

            img = TF.to_pil_image(img_vis[0])
            prb = TF.to_pil_image(prob[0, 0].detach().cpu())
            prd = TF.to_pil_image((prob[0, 0].detach().cpu() >= ths[-1]).float())
            msk = TF.to_pil_image(y[0, 0].detach().cpu())

            (od / "viz").mkdir(exist_ok=True)
            img.save(od / "viz" / f"{stem}_img.png")
            prb.save(od / "viz" / f"{stem}_prob.png")
            prd.save(od / "viz" / f"{stem}_pred.png")
            msk.save(od / "viz" / f"{stem}_gt.png")
            saved += 1

    # write results
    if args.th_sweep:
        lines = ["th\tdice\tiou\tn"]
        best = None
        for th in ths:
            n = max(sums[th]["n"], 1)
            d = sums[th]["dice"] / n
            j = sums[th]["iou"] / n
            lines.append(f"{th:.2f}\t{d:.6f}\t{j:.6f}\t{n}")
            if best is None or d > best[1]:
                best = (th, d, j)
        (od / "th_sweep.tsv").write_text("\n".join(lines))
        (od / "metrics.txt").write_text(
            f"Best@th={best[0]:.2f}\nDice={best[1]:.6f}\nIoU={best[2]:.6f}\nN={sums[best[0]]['n']}\n"
        )
        print(f"\nDone. Best Dice@{best[0]:.2f} = {best[1]:.4f}  IoU={best[2]:.4f}  N={sums[best[0]]['n']}")
    else:
        th = ths[0]
        n = max(sums[th]["n"], 1)
        d = sums[th]["dice"] / n
        j = sums[th]["iou"] / n
        (od / "metrics.txt").write_text(f"Dice={d:.6f}\nIoU={j:.6f}\nN={n}\n")
        print(f"\nDone. Dice={d:.4f} IoU={j:.4f} N={n}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--norm", type=str, default="none", choices=["none", "imagenet"], help="input normalization")
    ap.add_argument("--th_sweep", action="store_true", help="sweep thresholds 0.05~0.95")
    ap.add_argument("--base_ch", type=int, default=None, help="override base_ch (otherwise inferred from ckpt)")
    ap.add_argument("--vis_n", type=int, default=3, help="save N visualization samples")
    ap.add_argument("--img_size", type=int, default=256, help="resize to (img_size, img_size)")
    ap.add_argument("--load_mode", type=str, default="relaxed", choices=["relaxed", "legacy_raw"],
                    help="relaxed(remap to UNetTiny) or legacy_raw(strict, no remap)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
