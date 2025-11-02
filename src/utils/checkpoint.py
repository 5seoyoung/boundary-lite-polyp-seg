# src/utils/checkpoint.py
import os, torch, re
from typing import Dict, Any
# src/utils/checkpoint.py
import os
import torch

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def pack_ckpt(model, epoch: int, val_dice: float, extra=None):
    sd = model.state_dict()
    payload = {
        "model": sd,
        "epoch": epoch,
        "val_dice": float(val_dice),
    }
    if extra is not None:
        payload["extra"] = extra
    return payload

def save_ckpt(out_dir, payload, is_best: bool):
    last_p = os.path.join(out_dir, "last.pt")
    torch.save(payload, last_p)
    if is_best:
        best_p = os.path.join(out_dir, "best.pt")
        torch.save(payload, best_p)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pack_ckpt(model, epoch: int, val_dice: float, extra: Dict[str,Any]=None):
    return {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "val_dice": float(val_dice),
        "meta": extra or {}
    }

def save_ckpt(out_dir: str, payload: Dict[str,Any], is_best: bool):
    ensure_dir(out_dir)
    torch.save(payload, os.path.join(out_dir, "last.pt"))
    if is_best:
        torch.save(payload, os.path.join(out_dir, "best.pt"))

def _is_v1_unet_tiny(keys):
    # 구버전 ckpt 키 패턴 (enc1/dec*/bott/out ...)
    return any(k.startswith(("enc1.","enc2.","bott.","dec1.","dec2.","out","up1","up2")) for k in keys)

def _map_v1_to_v2(sd_v1: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
    """
    구버전(state_dict: enc1/enc2/bott/dec1/dec2/out, up1/up2)을
    현행 버전(state_dict: inc.*, down1.*, down2.*, bot.*, up1.*, up2.*, outc)으로 매핑.
    프로젝트 내부 구조에 맞춰 필요한 최소 매핑만 구현.
    """
    sd = {}
    # enc1 -> inc.net[0], inc.net[1]
    mapping = [
        ("enc1.0.weight", "inc.net.0.0.weight"),
        ("enc1.0.bias",   "inc.net.0.0.bias"),
        ("enc1.1.weight", "inc.net.0.1.weight"),
        ("enc1.1.bias",   "inc.net.0.1.bias"),
        ("enc1.1.running_mean", "inc.net.0.1.running_mean"),
        ("enc1.1.running_var",  "inc.net.0.1.running_var"),
        ("enc1.3.weight", "inc.net.1.0.weight"),
        ("enc1.3.bias",   "inc.net.1.0.bias"),
        ("enc1.4.weight", "inc.net.1.1.weight"),
        ("enc1.4.bias",   "inc.net.1.1.bias"),
        ("enc1.4.running_mean", "inc.net.1.1.running_mean"),
        ("enc1.4.running_var",  "inc.net.1.1.running_var"),
        # enc2 -> down1.conv.net[*]
        ("enc2.0.weight", "down1.conv.net.0.0.weight"),
        ("enc2.0.bias",   "down1.conv.net.0.0.bias"),
        ("enc2.1.weight", "down1.conv.net.0.1.weight"),
        ("enc2.1.bias",   "down1.conv.net.0.1.bias"),
        ("enc2.1.running_mean","down1.conv.net.0.1.running_mean"),
        ("enc2.1.running_var","down1.conv.net.0.1.running_var"),
        ("enc2.3.weight", "down1.conv.net.1.0.weight"),
        ("enc2.3.bias",   "down1.conv.net.1.0.bias"),
        ("enc2.4.weight", "down1.conv.net.1.1.weight"),
        ("enc2.4.bias",   "down1.conv.net.1.1.bias"),
        ("enc2.4.running_mean","down1.conv.net.1.1.running_mean"),
        ("enc2.4.running_var","down1.conv.net.1.1.running_var"),
        # bott -> bot.net[*]
        ("bott.0.weight", "bot.net.0.0.weight"),
        ("bott.0.bias",   "bot.net.0.0.bias"),
        ("bott.1.weight", "bot.net.0.1.weight"),
        ("bott.1.bias",   "bot.net.0.1.bias"),
        ("bott.1.running_mean","bot.net.0.1.running_mean"),
        ("bott.1.running_var","bot.net.0.1.running_var"),
        ("bott.3.weight", "bot.net.1.0.weight"),
        ("bott.3.bias",   "bot.net.1.0.bias"),
        ("bott.4.weight", "bot.net.1.1.weight"),
        ("bott.4.bias",   "bot.net.1.1.bias"),
        ("bott.4.running_mean","bot.net.1.1.running_mean"),
        ("bott.4.running_var","bot.net.1.1.running_var"),
        # dec2 -> up1.conv.net[*]  (skip 연결 순서 차이는 conv가 흡수)
        ("dec2.0.weight", "up1.conv.net.0.0.weight"),
        ("dec2.0.bias",   "up1.conv.net.0.0.bias"),
        ("dec2.1.weight", "up1.conv.net.0.1.weight"),
        ("dec2.1.bias",   "up1.conv.net.0.1.bias"),
        ("dec2.1.running_mean","up1.conv.net.0.1.running_mean"),
        ("dec2.1.running_var","up1.conv.net.0.1.running_var"),
        ("dec2.3.weight", "up1.conv.net.1.0.weight"),
        ("dec2.3.bias",   "up1.conv.net.1.0.bias"),
        ("dec2.4.weight", "up1.conv.net.1.1.weight"),
        ("dec2.4.bias",   "up1.conv.net.1.1.bias"),
        ("dec2.4.running_mean","up1.conv.net.1.1.running_mean"),
        ("dec2.4.running_var","up1.conv.net.1.1.running_var"),
        # dec1 -> up2.conv.net[*]
        ("dec1.0.weight", "up2.conv.net.0.0.weight"),
        ("dec1.0.bias",   "up2.conv.net.0.0.bias"),
        ("dec1.1.weight", "up2.conv.net.0.1.weight"),
        ("dec1.1.bias",   "up2.conv.net.0.1.bias"),
        ("dec1.1.running_mean","up2.conv.net.0.1.running_mean"),
        ("dec1.1.running_var","up2.conv.net.0.1.running_var"),
        ("dec1.3.weight", "up2.conv.net.1.0.weight"),
        ("dec1.3.bias",   "up2.conv.net.1.0.bias"),
        ("dec1.4.weight", "up2.conv.net.1.1.weight"),
        ("dec1.4.bias",   "up2.conv.net.1.1.bias"),
        ("dec1.4.running_mean","up2.conv.net.1.1.running_mean"),
        ("dec1.4.running_var","up2.conv.net.1.1.running_var"),
        # out -> outc
        ("out.weight", "outc.weight"),
        ("out.bias",   "outc.bias"),
    ]
    for k_old, k_new in mapping:
        if k_old in sd_v1:
            sd[k_new] = sd_v1[k_old]
    return sd

def load_ckpt_compat(ckpt_path: str):
    blob = torch.load(ckpt_path, map_location="cpu")
    sd = blob.get("model", blob)
    keys = list(sd.keys())
    if _is_v1_unet_tiny(keys):
        sd = _map_v1_to_v2(sd)
        blob["__compat_mapped__"] = True
    else:
        blob["__compat_mapped__"] = False
    blob["model"] = sd
    return blob
