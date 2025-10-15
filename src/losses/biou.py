import torch
import torch.nn.functional as F

def _dilate(x, k):  return F.max_pool2d(x, kernel_size=2*k+1, stride=1, padding=k)
def _erode(x, k):   return -F.max_pool2d(-x, kernel_size=2*k+1, stride=1, padding=k)

def boundary_band(x, delta=3):
    # x: (B,1,H,W), [0..1]
    dil = _dilate(x, delta)
    ero = _erode(x, delta)
    return (dil - ero).clamp(0.0, 1.0)     # [0,1]

def biou_loss(logits: torch.Tensor, gt: torch.Tensor, delta=3, eps=1e-6):
    p  = torch.sigmoid(logits)
    gb = boundary_band(gt, delta)          # {0,1}
    pb = boundary_band(p,  delta)          # [0,1]
    inter = (pb*gb).sum(dim=(1,2,3))
    union = (pb + gb - pb*gb).sum(dim=(1,2,3))
    biou  = (inter + eps) / (union + eps)
    return 1.0 - biou.mean()
