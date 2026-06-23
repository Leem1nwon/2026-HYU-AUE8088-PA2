"""Sweep ViT Grad-CAM target modules to pick the most informative one.

For each candidate target we measure, over the showcase val images:
  - head-divergence : mean pairwise |norm CAM_i - CAM_j| over the 3 heads
                      (higher = the 3 task heads attend to more different regions)
  - focus           : 1 - mean(normCAM)  (higher = more localized, less diffuse)

Candidates: block OUTPUTS at -2..-6, and input-side norm1 of the last blocks
(norm1 = input to that block's attention -> nonzero patch gradient even at -1,
the standard pytorch-grad-cam ViT target).

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/level4_vit_cam_sweep.py
"""
from __future__ import annotations

import numpy as np
import torch

from src.datasets.bdd_attr import ATTRIBUTES, WEATHER_CLASSES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.seed import set_seed
from src.utils.transforms import eval_transform
from src.xai.gradcam import ViTGradCAM

CKPT = "checkpoints/level3_best.pth"
SHOWCASE = ["clear", "rainy", "snowy", "overcast"]


def head_div(cams):
    n = [c for c in cams]                       # already [0,1]
    pairs = [(0, 1), (0, 2), (1, 2)]
    return float(np.mean([np.abs(n[i] - n[j]).mean() for i, j in pairs]))


def pick_images(ds):
    want = {WEATHER_CLASSES.index(c) for c in SHOWCASE}
    chosen, seen = [], set()
    for i in range(len(ds)):
        w = ds.samples[i].weather
        if w in want and w not in seen:
            seen.add(w); chosen.append(i)
        if len(seen) == len(want):
            break
    return chosen


def main():
    set_seed(42, deterministic=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_small_patch16_224().to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    model.float().to(dev).eval()
    ds = BDDAttrDataset("data/set_a", "val", transform=eval_transform())
    idxs = pick_images(ds)

    nb = len(model.blocks)
    candidates = {
        "blocks[-2] out": model.blocks[nb - 2],
        "blocks[-3] out": model.blocks[nb - 3],
        "blocks[-4] out": model.blocks[nb - 4],
        "blocks[-5] out": model.blocks[nb - 5],
        "blocks[-6] out": model.blocks[nb - 6],
        "blocks[-1].norm1": model.blocks[nb - 1].norm1,
        "blocks[-2].norm1": model.blocks[nb - 2].norm1,
        "blocks[-3].norm1": model.blocks[nb - 3].norm1,
    }

    print(f"{'target':18s} {'head-div':>9s} {'focus':>7s}")
    results = {}
    for name, mod in candidates.items():
        cam_fn = ViTGradCAM(model, mod)
        divs, focus = [], []
        for idx in idxs:
            x = ds[idx]["image"].unsqueeze(0).to(dev)
            cams = [cam_fn(x, (lambda a: (lambda out: out[a].max()))(at))[0].cpu().numpy()
                    for at in ATTRIBUTES]
            divs.append(head_div(cams))
            focus.append(1.0 - float(np.mean([c.mean() for c in cams])))
        d, f = float(np.mean(divs)), float(np.mean(focus))
        results[name] = (d, f)
        print(f"{name:18s} {d:9.3f} {f:7.3f}")

    best = max(results, key=lambda k: results[k][0])
    print(f"\nhead-divergence 최대: {best} (div={results[best][0]:.3f}, focus={results[best][1]:.3f})")


if __name__ == "__main__":
    main()
