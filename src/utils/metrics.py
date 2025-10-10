import torch
def dice_coeff(pred, target, eps=1e-6):
    # pred: probs in [0,1], target: {0,1}
    inter = (pred*target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + eps) / (union + eps)
