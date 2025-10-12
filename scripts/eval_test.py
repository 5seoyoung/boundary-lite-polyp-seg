import os, argparse, yaml, numpy as np
import torch
from glob import glob
from tqdm import tqdm

from src.models.unet_tiny import UNetTiny
from src.data.dataset import list_pairs, read_image, read_mask, overlay

def load_yaml(p): 
    with open(p) as f: return yaml.safe_load(f)

def choose_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def bin_metrics(pred_bin, gt_bin, eps=1e-6):
    inter = (pred_bin & gt_bin).sum()
    a = pred_bin.sum(); b = gt_bin.sum()
    dice = (2*inter + eps) / (a + b + eps)
    union = a + b - inter
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_data", default="configs/data.yaml")
    ap.add_argument("--cfg_train", default="configs/train_baseline.yaml")
    ap.add_argument("--ckpt", default="outputs/baseline/best.pt")
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--save_samples", type=int, default=20)
    ap.add_argument("--out_dir", default="outputs/test_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg_data  = load_yaml(args.cfg_data)
    cfg_train = load_yaml(args.cfg_train)

    setting = cfg_data["splits"]["setting"]
    if setting == "A":   # Kvasir -> test CVC
        test_root = cfg_data["cvc"]["path"]
    elif setting == "B": # CVC -> test Kvasir
        test_root = cfg_data["kvasir"]["path"]
    else:                # C: in-domain (간단히 kvasir로)
        test_root = cfg_data["kvasir"]["path"]

    size = tuple(cfg_data["img_size"])
    device = choose_device()
    print(f"device: {device}  setting: {setting}  test_root: {test_root}")

    # 모델 로드
    model = UNetTiny(cfg_train["in_channels"], cfg_train["num_classes"]).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    pairs = list_pairs(test_root)
    if len(pairs) == 0:
        raise SystemExit(f"No test pairs found under {test_root}")

    dices, ious = [], []
    save_n = min(args.save_samples, len(pairs))
    for i, (im_p, m_p) in enumerate(tqdm(pairs, desc="Eval")):
        im = read_image(im_p, size)         # RGB 0..255
        gt = read_mask(m_p, size) > 0       # bool

        x = torch.from_numpy(im.transpose(2,0,1)).float().unsqueeze(0)/255.0
        x = x.to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0,0].cpu().numpy()

        pred = (prob >= args.th)
        d, j = bin_metrics(pred, gt)
        dices.append(d); ious.append(j)

        if i < save_n:
            import cv2
            vis = overlay(im, (pred.astype(np.uint8)*255), alpha=0.45)
            cv2.imwrite(os.path.join(args.out_dir, f"sample_{i:02d}.png"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    Dice = float(np.mean(dices)); IoU = float(np.mean(ious))
    print(f"Test Dice: {Dice:.4f}  IoU: {IoU:.4f}  (N={len(pairs)})")
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(f"Dice={Dice:.6f}\nIoU={IoU:.6f}\nN={len(pairs)}\n")

if __name__ == "__main__":
    main()
