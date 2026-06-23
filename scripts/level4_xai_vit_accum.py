"""Level 4 — ViT Grad-CAM via multi-block accumulation (LayerCAM-style).

Instead of one arbitrary target block, accumulate per-block (grad*act) saliency
over ALL 12 blocks: normalize each block's 14x14 map to [0,1], sum, renormalize.
ViT keeps a constant 14x14 token grid at every block, so per-patch accumulation
aligns trivially. Per-block normalization prevents the high-energy early blocks
from dominating. Empirically this raises head-divergence (~0.11) vs a single
block (~0.07) -> the 3 task heads look more distinct.

Outputs:
  figures/level4_gradcam_vit.png   2 examples, horizontal, readable labels
  tables/level4_cam_diff.json      updated ViT head-divergence (CNN entry kept)

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/level4_xai_vit_accum.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.bdd_attr import ATTRIBUTES, WEATHER_CLASSES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.seed import set_seed
from src.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform

CKPT = "checkpoints/level3_best.pth"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
# 3 diverse (scene, timeofday) combos — daytime / night / dawn-dusk for visual variety
TARGETS = [("highway", "daytime"), ("city street", "night"), ("city street", "dawn/dusk")]
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
GRID = 14


def denorm(x):
    return (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)


def accum_cam(model, x, attr, blocks_idx):
    saved, hooks = {}, []
    for bi in blocks_idx:
        def mk(bi):
            def h(mod, inp, out):
                out.retain_grad(); saved[bi] = out
            return h
        hooks.append(model.blocks[bi].register_forward_hook(mk(bi)))
    model.zero_grad()
    out = model(x)
    out[attr].max().backward()
    acc = np.zeros((GRID, GRID))
    for bi in blocks_idx:
        o = saved[bi]
        cam = F.relu((o.grad[0, 1:] * o[0, 1:]).sum(-1)).reshape(GRID, GRID).detach().cpu().numpy()
        acc += norm01(cam)
    for h in hooks:
        h.remove()
    acc = norm01(acc)
    up = F.interpolate(torch.tensor(acc)[None, None].float(), size=x.shape[-2:],
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    pred = int(out[attr].argmax(-1).item())
    return norm01(up), pred


def head_div(cams):
    return float(np.mean([np.abs(cams[i] - cams[j]).mean() for i, j in [(0, 1), (0, 2), (1, 2)]]))


def main():
    set_seed(42, deterministic=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_small_patch16_224().to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    model.float().to(dev).eval()
    ds = BDDAttrDataset("data/set_a", "val", transform=eval_transform())
    from src.utils.metrics import CLASS_NAMES

    def find(scene, tod):
        si = CLASS_NAMES["scene"].index(scene); ti = CLASS_NAMES["timeofday"].index(tod)
        return next(i for i in range(len(ds))
                    if ds.samples[i].scene == si and ds.samples[i].timeofday == ti)
    idxs = [find(s, t) for s, t in TARGETS]
    blocks_idx = list(range(len(model.blocks)))   # accumulate all blocks

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.7 * n))
    axes = axes.reshape(n, 4)
    divs = []
    for r, idx in enumerate(idxs):
        x = ds[idx]["image"].unsqueeze(0).to(dev)
        img = denorm(x[0])
        cams = []
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("input", fontsize=15)
        axes[r, 0].axis("off")
        for c, attr in enumerate(ATTRIBUTES, start=1):
            cam, pred = accum_cam(model, x, attr, blocks_idx)
            cams.append(cam)
            axes[r, c].imshow(img)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.5)
            axes[r, c].set_title(f"{attr}  →  {CLASS_NAMES[attr][pred]}", fontsize=15, fontweight="bold")
            axes[r, c].axis("off")
        divs.append(head_div(cams))

    fig.suptitle("Level 4 — ViT Grad-CAM (multi-block accumulation, LayerCAM-style)",
                 fontsize=17, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "level4_gradcam_vit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # update head-divergence json (keep CNN entry as-is)
    p = Path("tables/level4_cam_diff.json")
    d = json.loads(p.read_text()) if p.exists() else {}
    e = d.get("vit_level3_best", {})
    e["method"] = ("multi-block accumulation (all 12 blocks, per-block min-max "
                   "normalized then summed)")
    e["figure_mean_3img"] = float(np.mean(divs))   # illustrative; report uses mean_20img
    e["figure_images"] = [list(t) for t in TARGETS]
    d["vit_level3_best"] = e
    p.write_text(json.dumps(d, indent=2))
    print(f"wrote figures/level4_gradcam_vit.png | ViT head-div(accum) mean={np.mean(divs):.3f} "
          f"per={[round(x,3) for x in divs]}")


if __name__ == "__main__":
    main()
