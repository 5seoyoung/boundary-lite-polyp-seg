# scripts/eval_test.py
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import numpy as np
import torch, cv2
from PIL import Image
from tqdm import tqdm

from src.models.unet_tiny import UNetTiny
from src.utils.preproc import get_preproc

# --- optional eval.lock.yaml (log only) ------------------------------
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

_lock = {}
_lock_path = Path("configs/eval.lock.yaml")
if _lock_path.exists() and yaml is not None:
    try:
        _lock = yaml.safe_load(_lock_path.read_text()) or {}
        print("[eval-lock]", {k: _lock.get(k) for k in ("setting","img_size","preproc","device","test_name")})
    except Exception as e:
        print(f"[eval-lock] read failed: {e}")
# --------------------------------------------------------------------


def postprocess(mask_bin: np.ndarray, mode: str | None):
    if not mode: return mask_bin
    h, w = mask_bin.shape
    m = (mask_bin > 0).astype(np.uint8)
    if "min" in mode:
        # min area
        thr = int(mode.replace("min","").replace("_","").replace("fill","") or "80")
        cnt, labels = cv2.connectedComponents(m)
        out = np.zeros_like(m)
        for i in range(1, cnt):
            area = (labels==i).sum()
            if area >= thr: out[labels==i] = 1
        m = out
    if "fill" in mode:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=2)
    if "morph" in mode:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    return (m*255).astype(np.uint8)

def dice_iou(pred: np.ndarray, gt: np.ndarray):
    p = (pred>0).astype(np.uint8); g = (gt>0).astype(np.uint8)
    inter = (p & g).sum()
    d = (2*inter) / (p.sum() + g.sum() + 1e-6)
    i = inter / ((p|g).sum() + 1e-6)
    return d, i

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_root", default="data/CVC-ClinicDB")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--th", type=float, default=0.50)
    ap.add_argument("--preproc", default="none", choices=["none","clahe_y","color_norm"])
    ap.add_argument("--post", default=None, help="예: min80_fill, morph, fill")
    ap.add_argument("--save_samples", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNetTiny(in_ch=3, out_ch=1).to(device).eval()
    weights = torch.load(args.ckpt, map_location=device)["model"]
    net.load_state_dict(weights)

    preproc_fn = get_preproc(args.preproc)

    imgs = sorted((Path(args.test_root)/"images").glob("*"))
    msks = Path(args.test_root)/"masks"
    d_all, i_all = [], []
    saved = 0

    for ip in tqdm(imgs):
        gt = cv2.imread(str(msks/ip.name), cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        im = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        im = preproc_fn(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imr = cv2.resize(im, (args.img, args.img), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(imr).permute(2,0,1).float().unsqueeze(0)/255.0
        ten = ten.to(device)
        with torch.no_grad():
            pr = torch.sigmoid(net(ten))[0,0].cpu().numpy()
        pb = (pr >= args.th).astype(np.uint8)*255
        if args.post:
            pb = postprocess(pb, args.post)
        d, i = dice_iou(pb, cv2.resize(gt,(args.img,args.img), interpolation=cv2.INTER_NEAREST))
        d_all.append(d); i_all.append(i)

        if saved < args.save_samples:
            over = imr.copy()
            over[pb>0] = (0.6*over[pb>0] + 0.4*np.array([255,0,0])).astype(np.uint8)
            cv2.imwrite(str(Path(args.out_dir)/f"sample_{saved:03d}.jpg"),
                        cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            saved += 1

    md, mi = float(np.mean(d_all)), float(np.mean(i_all))
    with open(Path(args.out_dir)/"metrics.txt","w") as f:
        f.write(f"Dice {md:.4f}\nIoU {mi:.4f}\nN {len(d_all)}\n")
    print(f"RESULT  τ={args.th:.2f}  Dice={md:.4f}  IoU={mi:.4f}  N={len(d_all)}")

if __name__ == "__main__":
    main()
