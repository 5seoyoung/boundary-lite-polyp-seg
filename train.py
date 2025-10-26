import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os, yaml, random
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import list_pairs, read_image, read_mask
from src.models.unet_tiny import UNetTiny
from src.utils.metrics import dice_coeff

# ----------------------------
# helpers
# ----------------------------
def choose_device(mode='auto'):
    if mode=='auto':
        if torch.backends.mps.is_available(): return torch.device('mps')
        if torch.cuda.is_available(): return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(mode)

def load_yaml(path):
    with open(path) as f: return yaml.safe_load(f)

def make_split(pairs, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(pairs)
    n = len(pairs); v = int(n*val_ratio)
    return pairs[v:], pairs[:v]

def to_tensor(img, msk, size=(256,256), device='cpu'):
    im = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
    m  = torch.from_numpy((msk>0).astype(np.float32))[None, ...]
    return im.to(device), m.to(device)

# ---- 간단 증강 (옵션) ----
def augment_np(im):
    """im: HxWx3 uint8"""
    import cv2
    if random.random()<0.7:  # 밝기/대비
        alpha = 0.8 + 0.4*random.random()  # 0.8~1.2
        beta  = random.randint(-16,16)
        im = np.clip(alpha*im + beta, 0, 255).astype(np.uint8)
    if random.random()<0.5:  # 감마
        g = 0.8 + 0.8*random.random()      # 0.8~1.6
        im = np.clip(255*((im/255.0)**g),0,255).astype(np.uint8)
    if random.random()<0.4:  # 가우시안 블러
        k = random.choice([3,5])
        im = cv2.GaussianBlur(im,(k,k),0)
    if random.random()<0.4:  # 해상도 저하 후 복원
        h,w = im.shape[:2]
        scale = random.choice([0.75,0.6])
        im = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        im = cv2.resize(im, (w,h), interpolation=cv2.INTER_LINEAR)
    return im

def iter_loader(pairs, batch_size, size, device, training=False, use_aug=False):
    batch = []
    for (im_p, m_p) in pairs:
        im = read_image(im_p, size)
        m  = read_mask(m_p, size)
        if training and use_aug:
            im = augment_np(im)
        batch.append((im,m))
        if len(batch)==batch_size:
            ims = []; msks=[]
            for im,m in batch:
                ti, tm = to_tensor(im,m,size,device)
                ims.append(ti); msks.append(tm)
            yield torch.stack(ims), torch.stack(msks)
            batch = []
    if batch:
        ims = []; msks=[]
        for im,m in batch:
            ti, tm = to_tensor(im,m,size,device)
            ims.append(ti); msks.append(tm)
        yield torch.stack(ims), torch.stack(msks)

# ---- Baseline loss (fallback) ----
def bce_dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(probs, target)
    inter = (probs*target).sum(dim=(1,2,3))
    den   = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = 1 - ((2*inter + eps)/(den + eps)).mean()
    return bce + dice

# ---- Boundary-aware switch ----
def maybe_load_boundary(cfg_train_path):
    """
    Returns:
      use_boundary(bool), cfg_train(dict), fns(dict or None)
    """
    from pathlib import Path
    p = Path(cfg_train_path)
    if not p.exists():
        return False, None, None
    cfg = load_yaml(str(p))
    from src.losses.edge_weight import region_weighted_loss
    from src.losses.biou import biou_loss
    fns = {"region_weighted_loss": region_weighted_loss, "biou_loss": biou_loss}
    return True, cfg, fns

def getv(cfg, keys, default=None):
    """nested/flat config 호환 getter"""
    for k in keys:
        node = cfg
        ok = True
        for t in k.split('.'):
            if isinstance(node, dict) and t in node:
                node = node[t]
            else:
                ok = False; break
        if ok: return node
    return default

def main():
    # ---------------- cfg load ----------------
    cfg_data = load_yaml("configs/data.yaml")

    use_boundary, cfg_boundary, fns = maybe_load_boundary("configs/train_boundary.yaml")
    cfg_tr = load_yaml("configs/train_baseline.yaml")  # defaults
    if use_boundary:
        cfg_tr = {**cfg_tr, **cfg_boundary}

    # ---------------- dataset routing ----------------
    setting = cfg_data["splits"]["setting"]
    if setting == "A":
        train_root = cfg_data["kvasir"]["path"];  _ = cfg_data["cvc"]["path"]
    elif setting == "B":
        train_root = cfg_data["cvc"]["path"];     _ = cfg_data["kvasir"]["path"]
    else:
        train_root = cfg_data["kvasir"]["path"]

    # split
    train_pairs_all = list_pairs(train_root)
    train_pairs, val_pairs = make_split(
        train_pairs_all,
        cfg_data["splits"]["val_ratio"],
        cfg_data["splits"]["seed"]
    )
    print("train:", len(train_pairs), "val:", len(val_pairs))

    # ---------------- device / model / opt ----------------
    device = choose_device(getv(cfg_tr, ["device"], "auto"))
    print("device:", device)

    in_ch  = getv(cfg_tr, ["in_channels"], 3)
    n_cls  = getv(cfg_tr, ["num_classes"], 1)
    model = UNetTiny(in_ch, n_cls).to(device)

    lr  = float(getv(cfg_tr, ["optimizer.lr", "lr"], 3e-4))
    wd  = float(getv(cfg_tr, ["optimizer.weight_decay", "weight_decay"], 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    out_dir = getv(cfg_tr, ["out_dir"], "outputs/baseline")
    os.makedirs(out_dir, exist_ok=True)

    max_epochs = int(getv(cfg_tr, ["scheduler.max_epochs", "epochs"], 80))
    batch_size = int(getv(cfg_tr, ["scheduler.batch_size", "batch_size"], 16))
    size = tuple(cfg_data["img_size"])
    use_aug = bool(getv(cfg_tr, ["augment"], False))

    # boundary loss cfg
    if use_boundary:
        lambda_r = float(getv(cfg_tr, ["lambda_r"], 0.6))
        lambda_b = float(getv(cfg_tr, ["lambda_b"], 0.25))
        region_cfg = getv(cfg_tr, ["region"], {}) or {}
        biou_cfg   = getv(cfg_tr, ["biou"], {}) or {}
        print(">> Boundary-aware training ON  |  lambda_r=", lambda_r, " lambda_b=", lambda_b,
              " region:", region_cfg, " biou:", biou_cfg, f" | augment={use_aug}")
    else:
        print(">> Baseline loss (Dice+BCE)", f" | augment={use_aug}")

    # ---------------- train loop ----------------
    best_val = 0.0
    for epoch in range(1, max_epochs+1):
        model.train()
        losses=[]
        for x,y in iter_loader(train_pairs, batch_size, size, device, training=True, use_aug=use_aug):
            logits = model(x)

            if use_boundary:
                Lr = fns["region_weighted_loss"](
                    logits, y,
                    alpha=region_cfg.get("alpha", 2.0),
                    sigma=region_cfg.get("sigma", 3.0),
                    lambda_dice=region_cfg.get("dice_weight", 0.5),
                    lambda_bce =region_cfg.get("bce_weight", 0.5)
                )
                Lb = fns["biou_loss"](logits, y, delta=biou_cfg.get("delta", 3))
                loss = lambda_r * Lr + lambda_b * Lb
            else:
                loss = bce_dice_loss(logits, y)

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        tr_loss = float(np.mean(losses)) if losses else float("nan")

        # ---- val ----
        model.eval()
        dices=[]
        with torch.no_grad():
            for x,y in iter_loader(val_pairs, batch_size, size, device, training=False, use_aug=False):
                p = torch.sigmoid(model(x))
                dices.append(dice_coeff(p, y).item())
        val_dice = float(np.mean(dices)) if dices else 0.0

        print(f"[{epoch:03d}] train_loss={tr_loss:.4f}  val_dice={val_dice:.4f}")

        # save best
        if val_dice > best_val:
            best_val = val_dice
            ckpt = os.path.join(out_dir, "best.pt")
            torch.save({"model":model.state_dict(), "val_dice":best_val}, ckpt)

    print("best val dice:", best_val)

if __name__ == "__main__":
    main()
