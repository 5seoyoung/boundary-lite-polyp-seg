import os, yaml, random
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from src.data.dataset import list_pairs, read_image, read_mask
from src.models.unet_tiny import UNetTiny
from src.utils.metrics import dice_coeff

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
    n = len(pairs)
    v = int(n*val_ratio)
    return pairs[v:], pairs[:v]

def to_tensor(img, msk, size=(256,256), device='cpu'):
    im = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
    m  = torch.from_numpy((msk>0).astype(np.float32))[None, ...]
    return im.to(device), m.to(device)

def iter_loader(pairs, batch_size, size, device):
    batch = []
    for (im_p, m_p) in pairs:
        im = read_image(im_p, size)
        m  = read_mask(m_p, size)
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

def bce_dice_loss(logits, target):
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(probs, target)
    dice = 1 - dice_coeff(probs, target)
    return bce + dice

def main():
    cfg_data = load_yaml("configs/data.yaml")
    cfg_tr   = load_yaml("configs/train_baseline.yaml")

    # dataset routing
    setting = cfg_data["splits"]["setting"]
    if setting == "A":
        train_root = cfg_data["kvasir"]["path"]
        test_root  = cfg_data["cvc"]["path"]
    elif setting == "B":
        train_root = cfg_data["cvc"]["path"]
        test_root  = cfg_data["kvasir"]["path"]
    else:
        train_root = cfg_data["kvasir"]["path"]  # simple default
        test_root  = cfg_data["kvasir"]["path"]

    train_pairs = list_pairs(train_root)
    train_pairs, val_pairs = make_split(train_pairs,
                        cfg_data["splits"]["val_ratio"],
                        cfg_data["splits"]["seed"])
    print("train:", len(train_pairs), "val:", len(val_pairs))

    device = choose_device(cfg_tr["device"])
    print("device:", device)

    model = UNetTiny(cfg_tr["in_channels"], cfg_tr["num_classes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg_tr["optimizer"]["lr"],
                            weight_decay=cfg_tr["optimizer"]["weight_decay"])

    os.makedirs(cfg_tr["out_dir"], exist_ok=True)

    max_epochs = cfg_tr["scheduler"]["max_epochs"]
    batch_size = cfg_tr["scheduler"]["batch_size"]
    size = tuple(cfg_data["img_size"])

    best_val = 0.0
    for epoch in range(1, max_epochs+1):
        model.train()
        losses=[]
        for x,y in iter_loader(train_pairs, batch_size, size, device):
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tr_loss = np.mean(losses)

        # val
        model.eval()
        dices=[]
        with torch.no_grad():
            for x,y in iter_loader(val_pairs, batch_size, size, device):
                p = torch.sigmoid(model(x))
                dices.append(dice_coeff(p, y).item())
        val_dice = float(np.mean(dices))

        print(f"[{epoch:03d}] train_loss={tr_loss:.4f}  val_dice={val_dice:.4f}")

        # save best
        if val_dice > best_val:
            best_val = val_dice
            ckpt = os.path.join(cfg_tr["out_dir"], "best.pt")
            torch.save({"model":model.state_dict(), "val_dice":best_val}, ckpt)

    print("best val dice:", best_val)

if __name__ == "__main__":
    main()
