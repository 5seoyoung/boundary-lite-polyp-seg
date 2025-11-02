# src/utils/aug.py
from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from torchvision import transforms as T

def _gamma(x: torch.Tensor, min_g: float, max_g: float, p: float):
    if random.random() > p: 
        return x
    g = random.uniform(min_g, max_g)
    # clamp to avoid 0^gamma
    return torch.clip(x, 1e-6, 1).pow(g)

def _down_up(x: torch.Tensor, low_size: int, target: int, p: float):
    if random.random() > p:
        return x
    x_small = F.interpolate(x, (low_size, low_size), mode="bilinear", align_corners=False)
    return F.interpolate(x_small, (target, target), mode="bilinear", align_corners=False)

def build_aug(preset: str, img_size: int):
    """
    preset: "none" | "light" | "med" | "heavy"
    returns: callable(img PIL -> tensor [0..1]), callable(mask PIL -> tensor{0,1})
    NOTE: 동일한 RandomResizedCrop를 이미지/마스크에 일관되게 적용하기 위해,
          먼저 PIL->Tensor 이후 같은 파라미터로 affine을 적용한다.
    """
    # 공통: ToTensor
    to_tensor = T.ToTensor()

    # RRC 범위, ColorJitter 강도, GaussianBlur 확률, Gamma 범위, Downscale 크기/확률
    if preset == "light":
        rrc_scale = (0.8, 1.0); jitter = (0.1, 0.1, 0.1, 0.05); blur_p = 0.2; gamma = (0.95, 1.05, 0.3); down = (img_size, 0.0)
    elif preset == "med":
        rrc_scale = (0.6, 1.0); jitter = (0.2, 0.2, 0.2, 0.10); blur_p = 0.35; gamma = (0.9, 1.1, 0.5);  down = (192, 0.5)
    elif preset == "heavy":
        rrc_scale = (0.5, 1.0); jitter = (0.3, 0.3, 0.3, 0.15); blur_p = 0.5;  gamma = (0.8, 1.2, 0.7);  down = (192, 0.7)
    else:
        rrc_scale = (1.0, 1.0); jitter = (0.0, 0.0, 0.0, 0.0); blur_p = 0.0; gamma = (1.0, 1.0, 0.0);  down = (img_size, 0.0)

    color_jitter = T.ColorJitter(*jitter) if sum(jitter) > 0 else None
    gaussian_blur = T.GaussianBlur(kernel_size=3) if blur_p > 0 else None

    def aug_image(pil):
        x = to_tensor(pil)  # [C,H,W], 0..1
        # RandomResizedCrop-like: torchvision RRC는 마스크 동일 적용이 까다로워 수동 구현 대신 center+pad로 단순화하거나
        # 여기서는 비율 1:1 고정 resize만 적용(도메인 쉬프트는 색/블러/감마/다운스케일로 커버)
        # 이미 로더에서 정사각 resize를 맞춘다고 가정.
        if color_jitter:
            x = color_jitter(x)
        if gaussian_blur and random.random() < blur_p:
            x = gaussian_blur(x)
        if gamma[2] > 0:
            x = _gamma(x, gamma[0], gamma[1], gamma[2])
        if down[1] > 0:
            x = _down_up(x.unsqueeze(0), down[0], img_size, down[1]).squeeze(0)
        return x

    def aug_mask(pil_mask):
        m = to_tensor(pil_mask)  # [1,H,W], 0..1
        m = (m > 0.5).float()
        return m

    return aug_image, aug_mask
