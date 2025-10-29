#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.utils.metrics import dice_iou_batch

# ------------------------------------------------
# Data (Kvasir train/val from prepared folders)
# ------------------------------------------------
class SimplePairDataset(Dataset):
    def __init__(self, img_dir, msk_dir, img_size=(256,256), aug=None):
        self.img_dir = Path(img_dir)
        self.msk_dir = Path(msk_dir)
        imgs = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        msks = sorted([p for p in self.msk_dir.glob("*") if p.suffix.lower() in [".jpg",".png"]])
        cmn = sorted(list({p.stem for p in imgs} & {p.stem for p in msks}))
        self.pairs = [(self.img_dir/f"{s}{(self._ext(self.img_dir, s))}",
                       self.msk_dir/f"{s}{(self._ext(self.msk_dir, s, mask=True))}") for s in cmn]
        assert len(self.pairs)>0, f"no pairs in {img_dir} & {msk_dir}"
        self.img_size = img_size
        self.aug = aug

    def _ext(self, d, stem, mask=False):
        for e in [".png",".jpg",".jpeg"]:
            if (d/f"{stem}{e}").exists():
                return e
        # fallback
        return ".png" if mask else ".jpg"

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")

        # basic resize
        img = img.resize(self.img_size, Image.BILINEAR)
        msk = msk.resize(self.img_size, Image.NEAREST)

        # light augment (from --aug string)
        if self.aug:
            img = apply_aug(img, self.aug)

        img_t = TF.to_tensor(img)             # (3,H,W)
        msk_t = (torch.from_numpy((np.array(msk) > 127).astype(np.float32)))[None]
        return img_t, msk_t

# ------------------------------------------------
# Augmentation parser & apply
# ------------------------------------------------
def parse_aug(s: str):
    """
    example: rrc:0.6-1.0,jitter:0.2,gamma:0.9-1.1,jpeg:60-100
    """
    if not s: return {}
    out = {}
    for token in s.split(","):
        if ":" not in token: continue
        k, v = token.split(":", 1)
        if "-" in v:
            a, b = v.split("-", 1)
            try:
                a = float(a); b = float(b)
            except:
                a = int(a); b = int(b)
            out[k] = (a, b)
        else:
            try:
                out[k] = float(v)
            except:
                out[k] = int(v)
    return out

def apply_aug(img: Image.Image, cfg: dict) -> Image.Image:
    # random resized crop
    if "rrc" in cfg:
        lo, hi = cfg["rrc"]
        scale = random.uniform(lo, hi)
        w, h = img.size
        nw, nh = int(w*scale), int(h*scale)
        nw = max(32, min(nw, w)); nh = max(32, min(nh, h))
        x0 = random.randint(0, w-nw) if w>nw else 0
        y0 = random.randint(0, h-nh) if h>nh else 0
        img = img.crop((x0, y0, x0+nw, y0+nh)).resize((w,h), Image.BILINEAR)

    if "jitter" in cfg:
        j = cfg["jitter"]
        # brightness/contrast/saturation hue small shim
        b = 1.0 + random.uniform(-j, j)
        c = 1.0 + random.uniform(-j, j)
        img = ImageOps.autocontrast(img)
        img = ImageEnhanceBrightness(img, b)
        img = ImageEnhanceContrast(img, c)

    if "gamma" in cfg:
        g0, g1 = cfg["gamma"]
        g = random.uniform(g0, g1)
        img = ImageOps.autocontrast(img)
        img = img.point(lambda x: 255.0 * ((x/255.0) ** (1.0/g)))

    if "jpeg" in cfg:
        q0, q1 = cfg["jpeg"]
        q = int(random.uniform(q0, q1))
        # simulate jpeg compression in-memory
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img

# tiny helpers: brightness/contrast (PIL-lite)
def ImageEnhanceBrightness(img: Image.Image, factor: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def ImageEnhanceContrast(img: Image.Image, factor: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    mean = arr.mean(axis=(0,1), keepdims=True)
    arr = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ------------------------------------------------
# Model
# ------------------------------------------------
def build_model(device):
    err = None
    try:
        from src.models.unet_tiny import UNetTiny
        try:
            m = UNetTiny(in_channels=3, out_channels=1)
        except TypeError:
            m = UNetTiny(in_ch=3, out_ch=1)
        return m.to(device)
    except Exception as e:
        err = e
    try:
        from src.models.unet_tiny import LegacyUNetTiny
        try:
            m = LegacyUNetTiny(in_channels=3, out_channels=1)
        except TypeError:
            m = LegacyUNetTiny(in_ch=3, out_ch=1)
        return m.to(device)
    except Exception:
        if err: raise err
        raise

# ------------------------------------------------
# Loss (Dice + BCE)
# ------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.eps = eps
    def forward(self, logits, target):
        # BCE
        bce = self.bce(logits, target)
        # Dice
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1,2,3))
        den   = (prob + target).sum(dim=(1,2,3))
        dice = (2*inter + self.eps) / (den + self.eps)
        dice_loss = 1 - dice.mean()
        return self.w * bce + (1 - self.w) * dice_loss

# ------------------------------------------------
# Train / Val
# ------------------------------------------------
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="mps" else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Data (Setting A: train/val = Kvasir)
    img_root = Path("data/Kvasir-SEG/images")
    msk_root = Path("data/Kvasir-SEG/masks")
    assert img_root.exists() and msk_root.exists(), "Kvasir-SEG not prepared"

    # simple split
    all_stems = sorted(list({p.stem for p in img_root.glob("*")} & {p.stem for p in msk_root.glob("*")}))
    random.seed(0); random.shuffle(all_stems)
    n = len(all_stems)
    n_val = max(50, int(0.1 * n))
    val_stems = set(all_stems[:n_val])
    tr_stems  = set(all_stems[n_val:])

    def subset_loader(stems, bs, aug_cfg=None):
        def symlink_subset(tmpd, stems):
            (tmpd/"images").mkdir(parents=True, exist_ok=True)
            (tmpd/"masks").mkdir(parents=True, exist_ok=True)
            for s in stems:
                for e in [".jpg",".png",".jpeg"]:
                    ip = img_root/f"{s}{e}"
                    if ip.exists(): break
                mp = msk_root/f"{s}.png" if (msk_root/f"{s}.png").exists() else msk_root/f"{s}.jpg"
                (tmpd/"images"/f"{s}{ip.suffix}").symlink_to(ip.resolve())
                (tmpd/"masks"/f"{s}{mp.suffix}").symlink_to(mp.resolve())
        tmp = out / (".cache_split_" + ("train" if aug_cfg else "val"))
        if not tmp.exists():
            symlink_subset(tmp, stems)
        ds = SimplePairDataset(tmp/"images", tmp/"masks", img_size=(256,256), aug=aug_cfg)
        return DataLoader(ds, batch_size=bs, shuffle=True if aug_cfg else False, num_workers=0, pin_memory=False)

    aug_cfg = parse_aug(args.aug) if args.aug else None
    train_loader = subset_loader(tr_stems, args.batch_size, aug_cfg=aug_cfg)
    val_loader   = subset_loader(val_stems, 1, aug_cfg=None)

    model = build_model(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = DiceBCELoss(bce_weight=0.5)
    best = -1.0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", ncols=80)
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # val
        model.eval()
        vd, vj, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                d, j = dice_iou_batch(logits, y, th=0.5)
                vd += d; vj += j; n += 1
        vd /= max(n,1); vj /= max(n,1)

        print(f"[val] Dice={vd:.4f} IoU={vj:.4f} (N={n})")
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": vd, "val_iou": vj},
                   out/"last.pt")
        if vd > best:
            best = vd
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_dice": vd, "val_iou": vj},
                       out/"best.pt")
    print(f"done. best Dice={best:.4f}")

# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aug", type=str, default="")  # e.g. rrc:0.6-1.0,jitter:0.2,gamma:0.9-1.1,jpeg:60-100
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
