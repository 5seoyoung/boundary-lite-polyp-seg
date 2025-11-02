# scripts/make_viz_grid.py
from pathlib import Path
from PIL import Image

viz = Path("outputs/cvc_eval_legacy_raw/viz")
out = viz / "grids"; out.mkdir(exist_ok=True)

def grid(stem):
    img = Image.open(viz/f"{stem}_img.png").convert("RGB")
    gt  = Image.open(viz/f"{stem}_gt.png").convert("L").convert("RGB")
    prb = Image.open(viz/f"{stem}_prob.png").convert("L").convert("RGB")
    prd = Image.open(viz/f"{stem}_pred.png").convert("L").convert("RGB")
    w,h = img.size
    canvas = Image.new("RGB", (2*w, 2*h), (255,255,255))
    canvas.paste(img, (0,0))
    canvas.paste(gt,  (w,0))
    canvas.paste(prb, (0,h))
    canvas.paste(prd, (w,h))
    canvas.save(out/f"{stem}_grid.png")

for p in viz.glob("*_img.png"):
    grid(p.stem[:-4])  # remove "_img"
print("saved:", len(list(out.glob("*.png"))), "grids ->", out)
