import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os, random
from src.data.dataset import list_pairs, read_image, read_mask, overlay

def check(root, name, n_samples=5, size=(256,256), out_dir="./outputs/check"):
    os.makedirs(out_dir, exist_ok=True)
    pairs = list_pairs(root)
    print(f"[{name}] found pairs:", len(pairs))
    assert len(pairs) > 0, f"No pairs found in {root}"
    for i, (img, msk) in enumerate(random.sample(pairs, min(n_samples, len(pairs)))):
        im = read_image(img, size)
        m  = read_mask(msk, size)
        ov = overlay(im, m)
        out = os.path.join(out_dir, f"{name}_{i}.png")
        import cv2
        cv2.imwrite(out, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(f"Saved {min(n_samples, len(pairs))} overlays to {out_dir}")

if __name__ == "__main__":
    check("./data/Kvasir-SEG", "kvasir")
    check("./data/CVC-ClinicDB", "cvc")
    print("OK.")
