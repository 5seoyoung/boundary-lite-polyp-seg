# scripts/export_onnx.py
from __future__ import annotations
import argparse, torch, numpy as np, onnx
from pathlib import Path
from src.models.unet_tiny import UNetTiny

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net = UNetTiny(3,1)
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    net.load_state_dict(sd)
    net.eval()

    dummy = torch.zeros(1,3,args.img,args.img, dtype=torch.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17, dynamic_axes={"input":{0:"N"}, "logits":{0:"N"}}
    )
    onnx.load(str(out_path))  # quick check
    print(f"Saved ONNX → {out_path}")

if __name__ == "__main__":
    main()
