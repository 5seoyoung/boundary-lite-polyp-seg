#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, re, os, glob
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ---------- Dataset ----------
class CVCTestDataset(Dataset):
    def __init__(self, root="data/CVC-ClinicDB", img_size=256, norm="none"):
        self.root = Path(root)
        self.img_size = (img_size, img_size)
        self.norm = norm
        pats = [
            str(self.root / "images" / "*.png"),
            str(self.root / "Image" / "*.png"),
            str(self.root / "**" / "images" / "*.png"),
            str(self.root / "**" / "Image" / "*.png"),
        ]
        imgs = []
        for p in pats:
            imgs.extend(glob.glob(p, recursive=True))
        imgs = sorted(set(imgs))
        if not imgs:
            raise FileNotFoundError(f"No images under {self.root}. Expected {self.root}/images/*.png")
        self.imgs = imgs
        if norm == "imagenet":
            self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            self.std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        else:
            self.mean = None
            self.std = None

    def _mask_from_img(self, ip: str) -> str:
        mp = ip.replace("/images/", "/masks/").replace("/Image/", "/masks/")
        if os.path.exists(mp):
            return mp
        base, ext = os.path.splitext(mp)
        cand = base.replace("_mask", "") + "_mask" + ext
        if os.path.exists(cand):
            return cand
        masks_dir = mp.rsplit("/", 1)[0]
        fname = os.path.basename(ip)
        return os.path.join(masks_dir, fname)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = self._mask_from_img(ip)
        img = Image.open(ip).convert("RGB").resize(self.img_size, Image.BILINEAR)
        if not os.path.exists(mp):
            msk = Image.fromarray(np.zeros(self.img_size, dtype=np.uint8))
        else:
            msk = Image.open(mp).convert("L").resize(self.img_size, Image.NEAREST)
        x = TF.to_tensor(img)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        y = torch.from_numpy((np.array(msk) > 127).astype(np.float32))[None]
        stem = os.path.splitext(os.path.basename(ip))[0]
        return x, y, stem

# ---------- Model (legacy raw deconv UNet) ----------
def _conv3x3(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True),
    )

class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(_conv3x3(ic, oc), _conv3x3(oc, oc))
    def forward(self, x): return self.net(x)

class LegacyRawUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4  # 16,32,64
        self.inc   = DoubleConv(in_ch, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))   # 1/2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))   # 1/4
        self.bot   = DoubleConv(c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)    # 1/4->1/2
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)    # 1/2->1/1
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out = nn.Conv2d(c1, out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); xb = self.bot(x3)
        u2 = self.up2(xb); d2 = self.dec2(torch.cat([u2, x2], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, x1], 1))
        return self.out(d1)

# ---------- Remap (SAFE, fixed net.net bug) ----------
def remap_legacy(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map legacy keys:
      enc1./enc2./bott./dec1./dec2./out.  → inc./down1.1./bot./dec1./dec2./out.
    For DoubleConv blocks, insert a single '.net.' AFTER the block root (once),
    then remap tail indices conditionally:
      - if we injected '.net.': .0→0.0, .1→0.1, .3→1.0, .4→1.1
      - if we DID NOT inject:   .0→net.0.0, .1→net.0.1, .3→net.1.0, .4→net.1.1
    This prevents '...net.net...' from happening.
    """
    def prefix_map(k: str) -> str:
        k = k.replace(".block.", ".net.")  # harmless if absent
        k = re.sub(r"^enc1\.", "inc.", k)
        k = re.sub(r"^enc2\.", "down1.1.", k)  # down1 = [pool, DoubleConv] → index 1
        k = re.sub(r"^bott\.", "bot.", k)
        k = re.sub(r"^dec2\.", "dec2.", k)
        k = re.sub(r"^dec1\.", "dec1.", k)
        k = re.sub(r"^out\.",  "out.",  k)
        return k

    dcv_roots = ("inc", "bot", "dec1", "dec2", "down1.1", "down2.1")

    def has_net(k: str) -> bool:
        return ".net." in k

    def inject_net_once(k: str) -> tuple[str, bool]:
        # Insert '.net.' once right after the root, if not present yet.
        injected = False
        if any(k.startswith(root + ".") for root in dcv_roots) and not has_net(k):
            k = re.sub(r"^(inc|bot|dec1|dec2|down1\.1|down2\.1)\.", r"\1.net.", k, count=1)
            injected = True
        return k, injected

    def remap_tail(k: str, already_injected: bool) -> str:
        # If we already injected '.net.', DO NOT add another 'net.' in tail.
        # else, add 'net.' in tail positions.
        if already_injected or has_net(k):
            patterns = [("0", "0.0"), ("1", "0.1"), ("3", "1.0"), ("4", "1.1")]
        else:
            patterns = [("0", "net.0.0"), ("1", "net.0.1"), ("3", "net.1.0"), ("4", "net.1.1")]
        for a, b in patterns:
            k = re.sub(
                rf"\.{a}\.(weight|bias|running_mean|running_var|num_batches_tracked)$",
                rf".{b}.\1",
                k,
            )
        return k

    out = {}
    for k, v in sd.items():
        nk = prefix_map(k)
        nk, injected = inject_net_once(nk)
        nk = remap_tail(nk, injected)
        out[nk] = v
    return out

def load_legacy_non_strict(m: nn.Module, raw: Dict[str, torch.Tensor]) -> None:
    sd = raw.get("model", raw)
    sd = remap_legacy(sd)
    msd = m.state_dict()
    fit = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    missing = [k for k in msd.keys() if k not in fit]
    unexpected = [k for k in sd.keys() if k not in msd]
    print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:   print("  sample missing:", ", ".join(missing[:10]), "...")
    if unexpected:print("  sample unexpected:", ", ".join(unexpected[:10]), "...")
    m.load_state_dict(fit, strict=False)

# ---------- Post-proc ----------
def simple_postproc(pred01: torch.Tensor) -> torch.Tensor:
    try:
        import cv2
    except ImportError:
        return pred01
    m = pred01.detach().cpu().numpy().astype(np.uint8)*255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (labels == largest).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)
    return torch.from_numpy((m>127).astype(np.float32))

# ---------- Metrics ----------
def dice_iou(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

# ---------- Main ----------
def main(args):
    # device
    if args.device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
    else:
        dev = torch.device("cpu")

    # data
    ds = CVCTestDataset(root="data/CVC-ClinicDB", img_size=args.img_size, norm=args.norm)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # model + ckpt
    m = LegacyRawUNet(in_ch=3, out_ch=1, base=16).to(dev).eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    load_legacy_non_strict(m, sd)

    # out dirs
    od = Path(args.out_dir); (od/"viz").mkdir(parents=True, exist_ok=True)

    # thresholds
    ths = [round(t,2) for t in np.arange(0.05, 0.96, 0.05)] if args.th_sweep else [float(args.th)]
    sums = {th: {"dice":0.0, "iou":0.0, "n":0} for th in ths}

    saved = 0
    for x, y, stem in tqdm(loader, desc="eval", ncols=80):
        x, y = x.to(dev), y.to(dev)
        logits = m(x)
        # Calibration
        logits = logits * args.logit_gain + args.logit_bias
        prob = torch.sigmoid(logits)

        for th in ths:
            pred = (prob >= th).float()
            if args.postproc:
                pp = simple_postproc(pred[0,0])
                pred = pp[None,None].to(prob.device)
            d, j = dice_iou(pred, y)
            sums[th]["dice"] += d
            sums[th]["iou"]  += j
            sums[th]["n"]    += 1

        if saved < args.vis_n:
            img_vis = x.detach().cpu()
            if args.norm == "imagenet":
                mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
                std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
                img_vis = (img_vis*std + mean).clamp(0,1)
            img = TF.to_pil_image(img_vis[0])
            prb = TF.to_pil_image(prob[0,0].detach().cpu())
            prd = TF.to_pil_image((prob[0,0].detach().cpu()>=ths[-1]).float())
            msk = TF.to_pil_image(y[0,0].detach().cpu())
            img.save(od / "viz" / f"{stem}_img.png")
            prb.save(od / "viz" / f"{stem}_prob.png")
            prd.save(od / "viz" / f"{stem}_pred.png")
            msk.save(od / "viz" / f"{stem}_gt.png")
            saved += 1

    # write metrics
    if args.th_sweep:
        lines = ["th\tdice\tiou\tn"]; best = None
        for th in ths:
            n = max(sums[th]["n"],1)
            d = sums[th]["dice"]/n; j = sums[th]["iou"]/n
            lines.append(f"{th:.2f}\t{d:.6f}\t{j:.6f}\t{n}")
            if best is None or d > best[1]: best = (th, d, j)
        (od / "th_sweep.tsv").write_text("\n".join(lines))
        (od / "metrics.txt").write_text(
            f"Best@th={best[0]:.2f}\nDice={best[1]:.6f}\nIoU={best[2]:.6f}\nN={sums[best[0]]['n']}\n"
        )
        print(f"Done. Best Dice@{best[0]:.2f} = {best[1]:.4f}  IoU={best[2]:.4f}  N={sums[best[0]]['n']}")
    else:
        th = ths[0]; n = max(sums[th]["n"],1)
        d = sums[th]["dice"]/n; j = sums[th]["iou"]/n
        (od / "metrics.txt").write_text(f"Dice={d:.6f}\nIoU={j:.6f}\nN={n}\n")
        print(f"Done. Dice={d:.4f} IoU={j:.4f} N={n}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--norm", type=str, choices=["none","imagenet"], default="none")
    ap.add_argument("--th_sweep", action="store_true")
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--vis_n", type=int, default=3)
    # calibration & post-proc
    ap.add_argument("--logit_gain", type=float, default=1.0, help="multiply logits before sigmoid")
    ap.add_argument("--logit_bias", type=float, default=0.0, help="add bias to logits before sigmoid")
    ap.add_argument("--postproc", action="store_true", help="largest CC + closing")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
