#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm


# =========================
# Data loader (CVC-ClinicDB)
# =========================
def make_test_loader_cvc(root="data/CVC-ClinicDB", batch_size=1, img_size=(256, 256), norm="none"):
    """
    CVC-ClinicDB 평가용 로더.
    기대 구조:
      root/images/*.png
      root/masks/*.png   (파일명 동일)
    """
    from glob import glob
    import os

    if norm == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    else:
        mean = None
        std  = None

    class _DS(Dataset):
        def __init__(self, root, img_size, mean, std):
            self.root = Path(root)
            self.img_size = img_size
            self.mean = mean
            self.std = std

            img_dir = self.root / "images"
            msk_dir = self.root / "masks"
            assert img_dir.exists(), f"{img_dir} not found"
            assert msk_dir.exists(), f"{msk_dir} not found"

            imgs = sorted([Path(p) for p in glob(str(img_dir / "*.png"))])
            msks = sorted([Path(p) for p in glob(str(msk_dir / "*.png"))])
            img_stems = {p.stem for p in imgs}
            msk_stems = {p.stem for p in msks}
            commons = sorted(list(img_stems & msk_stems))
            assert len(commons) > 0, f"no image-mask pairs under {root}"

            self.pairs = [(img_dir / f"{s}.png", msk_dir / f"{s}.png") for s in commons]

        def __len__(self): return len(self.pairs)

        def __getitem__(self, i):
            ip, mp = self.pairs[i]
            img = Image.open(ip).convert("RGB").resize(self.img_size, Image.BILINEAR)
            msk = Image.open(mp).convert("L").resize(self.img_size, Image.NEAREST)

            x = TF.to_tensor(img)
            if self.mean is not None:
                x = (x - self.mean) / self.std
            y = torch.from_numpy((np.array(msk) > 127).astype(np.float32))[None]
            return x, y, ip.stem

    ds = _DS(root=root, img_size=img_size, mean=mean, std=std)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


# =========================
# Model import
# =========================
def _import_unet_tiny():
    from src.models.unet_tiny import UNetTiny
    return UNetTiny


# =========================
# Legacy → Current key remap
# =========================
# =========================
# Legacy → Current key remap (FIXED)
#  - enc1/enc2      → inc / down1.conv
#  - bott.0/1       → down2.conv.net.0.*   (32→64 첫 conv + BN)
#  - bott.3/4       → down2.conv.net.1.*   (64→64 둘째 conv + BN)
#  - dec2 / dec1    → up1.conv / up2.conv
#  - out            → outc
#  - .block.* index → .net.0.0 / .net.0.1 / .net.1.0 / .net.1.1
# =========================
def remap_legacy_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k

        # 1) 기본 rename
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
        # bott.* 는 아래에서 별도 분기 처리

        # 2) block → net
        nk = nk.replace(".block.", ".net.")

        # 3) 과거 인덱스 → net.*
        #   0 -> net.0.0, 1 -> net.0.1, 3 -> net.1.0, 4 -> net.1.1
        nk = nk.replace(".0.", ".net.0.0.")
        nk = nk.replace(".1.", ".net.0.1.")
        nk = nk.replace(".3.", ".net.1.0.")
        nk = nk.replace(".4.", ".net.1.1.")

        # 4) bott.* 분해 매핑:
        #    bott.0/1.*  == down2.conv.net.0.*
        #    bott.3/4.*  == down2.conv.net.1.*
        if k.startswith("bott.0.") or k.startswith("bott.1."):
            nk = re.sub(r"^bott\.(0|1)\.", "down2.conv.", nk, count=1)
            # 위에서 이미 .0./.1. → net.0.0./net.0.1. 로 치환됨
        elif k.startswith("bott.3.") or k.startswith("bott.4."):
            nk = re.sub(r"^bott\.(3|4)\.", "down2.conv.", nk, count=1)
            # 위에서 이미 .3./.4. → net.1.0./net.1.1. 로 치환됨
        elif k.startswith("bott."):
            # 혹시 모를 나머지(bn의 num_batches_tracked 등)는 down2 쪽으로 붙여줌
            nk = k.replace("bott.", "down2.conv.", 1)
            nk = nk.replace(".block.", ".net.")
            nk = nk.replace(".0.", ".net.0.0.").replace(".1.", ".net.0.1.")
            nk = nk.replace(".3.", ".net.1.0.").replace(".4.", ".net.1.1.")

        out[nk] = v
    return out


def infer_base_ch(sd: Dict[str, torch.Tensor], default: int = 16) -> int:
    for key in ("inc.net.0.0.weight", "inc.net.0.0.bias", "inc.net.0.1.weight", "enc1.block.0.weight", "enc1.0.weight"):
        if key in sd and sd[key].ndim >= 1:
            return int(sd[key].shape[0])
    return default


def is_legacy_deconv_ckpt(sd: Dict[str, torch.Tensor]) -> bool:
    """
    과거 포맷의 흔적: dec1/dec2/out 또는 up1/2 .weight/bias, 그리고 ConvTranspose2d 형태(2x2 kernel) 가시성 등.
    """
    keys = list(sd.keys())
    has_old_names = any(k.startswith(p) for k in keys for p in ("enc1.", "enc2.", "bott.", "dec1.", "dec2.", "out."))
    has_up_w = any(re.search(r"(up1|up2)\.(up|weight)", k) for k in keys)
    has_2x2 = any(sd[k].ndim == 4 and sd[k].shape[-2:] == (2, 2) for k in keys if k.endswith(".weight"))
    return has_old_names or has_up_w or has_2x2


# =========================
# Relaxed shape-fit loader
# =========================
def _shape_fit_copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor | None:
    """
    dst.shape 에 맞춰 src 를 crop/zero-pad 복사. 차원 수 다르면 None.
    """
    if dst.dim() != src.dim():
        return None
    out = dst.new_zeros(dst.shape)
    slices_dst, slices_src = [], []
    for d, s in zip(dst.shape, src.shape):
        m = min(d, s)
        slices_dst.append(slice(0, m))
        slices_src.append(slice(0, m))
    out[tuple(slices_dst)] = src[tuple(slices_src)].to(out.dtype)
    return out


def load_ckpt_relaxed(model: torch.nn.Module, sd_map: Dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    """
    remap된 sd_map을 model.state_dict()에 최대한 맞춰 로드.
    반환: (filled, skipped, resized)
    """
    msd = model.state_dict()
    new_sd: Dict[str, torch.Tensor] = {}
    filled, skipped, resized = [], [], []

    for k, dst_t in msd.items():
        if k not in sd_map:
            skipped.append(k)
            continue
        src_t = sd_map[k]
        if src_t.shape == dst_t.shape:
            new_sd[k] = src_t.to(dst_t.dtype)
            filled.append(k)
        else:
            fitted = _shape_fit_copy(dst_t, src_t)
            if fitted is None:
                skipped.append(k)
            else:
                new_sd[k] = fitted
                filled.append(k)
                resized.append(k)

    # strict=False 로 존재하는 키만 로드
    model.load_state_dict(new_sd, strict=False)
    return filled, skipped, resized


# =========================
# Metrics
# =========================
def dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


# =========================
# Eval
# =========================
@torch.no_grad()
def run_eval(args):
    # device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # ckpt load
    raw = torch.load(args.ckpt, map_location="cpu")
    sd_raw = raw.get("model", raw)

    # model build
    UNetTiny = _import_unet_tiny()
    base_ch = infer_base_ch(sd_raw, default=args.base_ch or 16)

    # 중요: 과거 ckpt라도 deconv 가 꼬여있는 경우가 많아 bilinear=True 강제 사용.
    # 업샘플 가중치는 무시되고 conv 계층만 최대한 로드(아래 relaxed 로더).
    bilinear = True
    if is_legacy_deconv_ckpt(sd_raw):
        print("[info] Legacy/deconv-like ckpt detected -> using UNetTiny(bilinear=True) with relaxed load")

    model = UNetTiny(in_channels=3, out_channels=1, base_ch=base_ch, bilinear=bilinear).to(device).eval()

    # remap & relaxed load
    sd_map = remap_legacy_keys(sd_raw)
    filled, skipped, resized = load_ckpt_relaxed(model, sd_map)
    print(f"[info] relaxed load: filled={len(filled)}, resized={len(resized)}, skipped={len(skipped)}")
    if resized:
        print("  resized sample:", ", ".join(resized[:10]), "...")
    if skipped:
        print("  skipped sample:", ", ".join(skipped[:10]), "...")

    # data
    H = W = int(args.img_size)
    loader = make_test_loader_cvc(root="data/CVC-ClinicDB",
                                  batch_size=1,
                                  img_size=(H, W),
                                  norm=args.norm)
    od = Path(args.out_dir)
    (od / "viz").mkdir(parents=True, exist_ok=True)

    # thresholds
    if args.th_sweep:
        ths = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    else:
        ths = [float(args.th)]

    sums = {th: {"dice": 0.0, "iou": 0.0, "n": 0} for th in ths}

    saved = 0
    for x, y, stem in tqdm(loader, desc="eval", ncols=80):
        x = x.to(device)
        y = y.to(device)
        prob = torch.sigmoid(model(x))  # (B,1,H,W)

        for th in ths:
            pred = (prob >= th).float()
            d, j = dice_iou(pred, y)
            sums[th]["dice"] += d
            sums[th]["iou"]  += j
            sums[th]["n"]    += 1

        if saved < args.vis_n:
            img_vis = x.detach().cpu()
            if args.norm == "imagenet":
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                img_vis = (img_vis * std + mean).clamp(0, 1)

            img = TF.to_pil_image(img_vis[0])
            prb = TF.to_pil_image(prob[0, 0].detach().cpu())
            prd = TF.to_pil_image((prob[0, 0].detach().cpu() >= ths[-1]).float())
            msk = TF.to_pil_image(y[0, 0].detach().cpu())

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


# =========================
# CLI
# =========================
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
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
