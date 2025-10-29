# -*- coding: utf-8 -*-
import torch

@torch.no_grad()
def dice_iou_batch(pred_logits: torch.Tensor, gt: torch.Tensor, th: float = 0.5, eps: float = 1e-7):
    """
    pred_logits: (B,1,H,W) float (로짓/확률 모두 허용)
    gt         : (B,1,H,W) {0,1}
    return     : (dice_mean, iou_mean) python float
    """
    if pred_logits.dtype.is_floating_point:
        pred_prob = torch.sigmoid(pred_logits)  # 이미 확률이어도 영향 미미
    else:
        pred_prob = pred_logits.float()

    pred = (pred_prob > th).float()
    gt = gt.float()

    dims = tuple(range(1, gt.ndim))
    inter = (pred * gt).sum(dims)
    union_d = pred.sum(dims) + gt.sum(dims)
    union_i = (pred + gt - pred * gt).sum(dims)

    dice = ((2 * inter + eps) / (union_d + eps)).mean()
    iou  = ((inter + eps) / (union_i + eps)).mean()
    return float(dice.item()), float(iou.item())
