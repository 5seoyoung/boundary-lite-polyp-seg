import torch
import torch.nn.functional as F
import cv2
import numpy as np

def _distance_to_edge(mask_np: np.ndarray) -> np.ndarray:
    """
    입력: mask_np (HxW, {0,255})
    출력: 경계까지의 거리 (float32)
    """
    m = (mask_np > 0).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m,  kernel, iterations=1)
    edge = (dil ^ ero).astype(np.uint8)           # 1 at boundary
    edge_inv = 1 - edge                           # 0 at boundary
    dist = cv2.distanceTransform(edge_inv, cv2.DIST_L2, 3)
    return dist.astype(np.float32)

def make_edge_weight(gt_mask: torch.Tensor, alpha=2.0, sigma=3.0) -> torch.Tensor:
    """
    gt_mask: (B,1,H,W), {0,1}
    W_edge = 1 + alpha * exp(-(d/sigma)^2)
    """
    B, _, H, W = gt_mask.shape
    weights = []
    for b in range(B):
        m = (gt_mask[b,0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        d = _distance_to_edge(m)
        w = 1.0 + alpha * np.exp(-(d / float(sigma))**2)
        weights.append(torch.from_numpy(w)[None,None,...])
    return torch.cat(weights, dim=0).to(gt_mask.device).float()

def dice_loss(logits: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor=None, eps=1e-6):
    p = torch.sigmoid(logits)
    if weight is None:
        inter = (p*gt).sum(dim=(1,2,3))
        den   = p.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3))
    else:
        inter = (weight*p*gt).sum(dim=(1,2,3))
        den   = (weight*p).sum(dim=(1,2,3)) + (weight*gt).sum(dim=(1,2,3))
    dice = (2*inter + eps) / (den + eps)
    return 1.0 - dice.mean()

def bce_loss(logits: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor=None):
    return F.binary_cross_entropy_with_logits(logits, gt, weight=weight)

def region_weighted_loss(logits, gt, alpha=2.0, sigma=3.0, lambda_dice=0.5, lambda_bce=0.5):
    W = make_edge_weight(gt, alpha=alpha, sigma=sigma)
    ld = dice_loss(logits, gt, weight=W)
    lb = bce_loss(logits, gt, weight=W)
    return lambda_dice*ld + lambda_bce*lb
