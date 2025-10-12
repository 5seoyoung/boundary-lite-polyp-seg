import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, argparse
import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import list_pairs, read_image, read_mask, overlay
from src.models.unet_tiny import UNetTiny

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--save_samples", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def choose_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def dice_iou(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    a = pred.sum(); b = gt.sum()
    dice = (2*inter + eps) / (a + b + eps)
    union = a + b - inter
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)

def main():
    import yaml
    cfg = yaml.safe_load(open("configs/data.yaml"))
    setting = cfg["splits"]["setting"]
    if setting == "A":
        test_root = cfg["cvc"]["path"]
    elif setting == "B":
        test_root = cfg["kvasir"]["path"]
    else:
        test_root = cfg["kvasir"]["path"]

    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = choose_device()

    # 모델 로드
    m = UNetTiny(3,1).to(dev)
    state = torch.load(args.ckpt, map_location=dev)
    m.load_state_dict(state["model"]); m.eval()

    pairs = list_pairs(test_root)
    print(f"device: {dev}  setting: {setting}  test_root: {test_root}")

    dices=[]; ious=[]
    save_n = int(args.save_samples)
    for i,(im_p, m_p) in enumerate(tqdm(pairs, desc="Eval")):
        im = read_image(im_p, tuple(cfg["img_size"]))
        gt = read_mask(m_p, tuple(cfg["img_size"])) > 0
        x = torch.from_numpy(im.transpose(2,0,1)).float().unsqueeze(0)/255.
        x = x.to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(m(x))[0,0].cpu().numpy()
        pred = prob >= args.th
        d,iou = dice_iou(pred, gt)
        dices.append(d); ious.append(iou)

        if save_n>0 and i<save_n:
            vis = overlay(im, (pred.astype(np.uint8)*255), alpha=0.45)
            outp = os.path.join(args.out_dir, f"sample_{i:03d}.png")
            from imageio.v2 import imwrite
            imwrite(outp, vis)

    D = float(np.mean(dices)); I = float(np.mean(ious)); N = len(pairs)
    print(f"Test Dice: {D:.4f}  IoU: {I:.4f}  (N={N})")
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(f"Dice={D:.6f}\nIoU={I:.6f}\nN={N}\n")

if __name__ == "__main__":
    main()
