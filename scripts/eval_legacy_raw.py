import argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

class CVC(Dataset):
    def __init__(self, root="data/CVC-ClinicDB", img_size=256, norm="imagenet"):
        r=Path(root); self.imgs=sorted((r/"images").glob("*.png")); self.msks=r/"masks"
        self.img_size=(img_size,img_size); self.norm=norm
        if norm=="imagenet":
            self.mean=torch.tensor([0.485,0.456,0.406])[:,None,None]
            self.std=torch.tensor([0.229,0.224,0.225])[:,None,None]
        else:
            self.mean=self.std=None
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        ip=self.imgs[i]; mp=self.msks/f"{ip.stem}.png"
        x=Image.open(ip).convert("RGB").resize(self.img_size, Image.BILINEAR)
        y=Image.open(mp).convert("L").resize(self.img_size, Image.NEAREST)
        x=TF.to_tensor(x); 
        if self.mean is not None: x=(x-self.mean)/self.std
        y=torch.from_numpy((np.array(y)>127).astype(np.float32))[None]
        return x,y,ip.stem

def dbl(in_ch,out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,out_ch,3,1,1, bias=True), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        nn.Conv2d(out_ch,out_ch,3,1,1, bias=True), nn.BatchNorm2d(out_ch), nn.ReLU(True),
    )

def _align(x, ref):
    # bilinear로 x를 ref의 (H,W)에 맞춘다
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

class LegacyRawUNet(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super().__init__()
        c1,c2,c3=16,32,64
        self.enc1=dbl(in_ch,c1)
        self.enc2=nn.Sequential(nn.MaxPool2d(2), dbl(c1,c2))
        self.bott=nn.Sequential(
            nn.Conv2d(c2,c3,3,1,1,bias=True), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.Conv2d(c3,c3,3,1,1,bias=True), nn.BatchNorm2d(c3), nn.ReLU(True),
        )
        self.up2=nn.ConvTranspose2d(c3,c2,2,2)
        self.dec2=dbl(c2+c2,c2)
        self.up1=nn.ConvTranspose2d(c2,c1,2,2)
        self.dec1=dbl(c1+c1,c1)
        self.out =nn.Conv2d(c1,out_ch,1)
    def forward(self,x):
        x1=self.enc1(x); x2=self.enc2(x1); xb=self.bott(x2)
        u2=self.up2(xb)
        # 크기 정렬 (u2: H, x2: H/2 인 케이스 대응)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = _align(u2, x2)
        u2=torch.cat([u2,x2],1)
        d2=self.dec2(u2)
        u1=self.up1(d2)
        # 크기 정렬 (u1: 2x, x1: x 일 수 있음)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = _align(u1, x1)
        u1=torch.cat([u1,x1],1)
        d1=self.dec1(u1)
        return self.out(d1)

def dice_iou(p,t,eps=1e-6):
    inter=(p*t).sum((1,2,3)); union=(p+t-p*t).sum((1,2,3))
    dice=(2*inter+eps)/(p.sum((1,2,3))+t.sum((1,2,3))+eps)
    iou=(inter+eps)/(union+eps)
    return dice.mean().item(), iou.mean().item()

def main(a):
    dev=torch.device("mps" if a.device=="mps" and torch.backends.mps.is_available() else "cpu")
    sd=torch.load(a.ckpt,map_location="cpu"); sd=sd.get("model",sd)
    m=LegacyRawUNet().to(dev).eval()
    miss,unexp=m.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] missing={len(miss)} unexpected={len(unexp)}")
    od=Path(a.out_dir); (od/"viz").mkdir(parents=True, exist_ok=True)
    loader=DataLoader(CVC(img_size=a.img_size, norm=a.norm), batch_size=1, shuffle=False)
    ths=[round(x,2) for x in np.arange(0.05,0.96,0.05)] if a.th_sweep else [a.th]
    sums={th:{"dice":0.0,"iou":0.0,"n":0} for th in ths}; saved=0
    with torch.no_grad():
        for x,y,stem in tqdm(loader,ncols=80,desc="eval"):
            x,y=x.to(dev),y.to(dev); prob=torch.sigmoid(m(x))
            for th in ths:
                pred=(prob>=th).float(); d,j=dice_iou(pred,y)
                sums[th]["dice"]+=d; sums[th]["iou"]+=j; sums[th]["n"]+=1
            if saved<a.vis_n:
                img=x.cpu(); 
                if a.norm=="imagenet":
                    mean=torch.tensor([0.485,0.456,0.406])[:,None,None]
                    std =torch.tensor([0.229,0.224,0.225])[:,None,None]
                    img=(img*std+mean).clamp(0,1)
                TF.to_pil_image(img[0]).save(od/"viz"/f"{stem}_img.png")
                TF.to_pil_image(prob[0,0].cpu()).save(od/"viz"/f"{stem}_prob.png")
                TF.to_pil_image((prob[0,0].cpu()>=ths[-1]).float()).save(od/"viz"/f"{stem}_pred.png")
                TF.to_pil_image(y[0,0].cpu()).save(od/"viz"/f"{stem}_gt.png"); saved+=1
    if a.th_sweep:
        best=None
        for th in ths:
            n=max(sums[th]["n"],1); d=sums[th]["dice"]/n; j=sums[th]["iou"]/n
            if best is None or d>best[1]: best=(th,d,j)
        print(f"Done. Best Dice@{best[0]:.2f} = {best[1]:.4f}  IoU={best[2]:.4f}  N={sums[best[0]]['n']}")
    else:
        th=ths[0]; n=max(sums[th]["n"],1); d=sums[th]["dice"]/n; j=sums[th]["iou"]/n
        print(f"Done. Dice={d:.4f} IoU={j:.4f} N={n}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--out_dir",required=True)
    ap.add_argument("--device",default="mps"); ap.add_argument("--norm",default="imagenet",choices=["none","imagenet"])
    ap.add_argument("--th_sweep",action="store_true"); ap.add_argument("--th",type=float,default=0.5)
    ap.add_argument("--img_size",type=int,default=256); ap.add_argument("--vis_n",type=int,default=5)
    main(ap.parse_args())
