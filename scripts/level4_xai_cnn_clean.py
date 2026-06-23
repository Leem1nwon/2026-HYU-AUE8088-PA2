"""Level 4 — ResNet-18 Grad-CAM, clean 2-example horizontal figure.

Standard Grad-CAM on the last conv block (layer4) — CNN's last conv keeps spatial
+ semantic info, so no dead-end issue (unlike ViT). Same 2 showcase images as the
ViT figure for a direct CNN-vs-ViT comparison. Large readable labels.

Output: figures/level4_gradcam_cnn.png  (+ updates CNN entry in level4_cam_diff.json)

Run:
  CUDA_VISIBLE_DEVICES=0 python scripts/level4_xai_cnn_clean.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.bdd_attr import ATTRIBUTES, WEATHER_CLASSES, BDDAttrDataset
from src.models.resnet import resnet18
from src.utils.metrics import CLASS_NAMES
from src.utils.seed import set_seed
from src.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.xai.gradcam import GradCAM

CKPT = "checkpoints/level1_resnet18.pth"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
# same 3 images as the ViT figure for a direct CNN-vs-ViT comparison
TARGETS = [("highway", "daytime"), ("city street", "night"), ("city street", "dawn/dusk")]
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denorm(x):
    return (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def norm01(a):
    a = np.asarray(a, np.float64); a = a - a.min()
    return a / (a.max() + 1e-8)


def head_div(cams):
    n = [norm01(c) for c in cams]
    return float(np.mean([np.abs(n[i] - n[j]).mean() for i, j in [(0, 1), (0, 2), (1, 2)]]))


def main():
    set_seed(42, deterministic=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18().to(dev)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    model.to(dev).eval()
    cam_fn = GradCAM(model, model.layer4[-1])
    ds = BDDAttrDataset("data/set_a", "val", transform=eval_transform())

    def find(scene, tod):
        si = CLASS_NAMES["scene"].index(scene); ti = CLASS_NAMES["timeofday"].index(tod)
        return next(i for i in range(len(ds))
                    if ds.samples[i].scene == si and ds.samples[i].timeofday == ti)
    idxs = [find(s, t) for s, t in TARGETS]

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.7 * n))
    axes = axes.reshape(n, 4)
    divs = []
    for r, idx in enumerate(idxs):
        x = ds[idx]["image"].unsqueeze(0).to(dev)
        img = denorm(x[0])
        with torch.no_grad():
            out0 = model(x)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("input", fontsize=15)
        axes[r, 0].axis("off")
        cams = []
        for c, attr in enumerate(ATTRIBUTES, start=1):
            cam = cam_fn(x, lambda out, a=attr: out[a].max(dim=-1).values.sum())[0].cpu().numpy()
            cams.append(cam)
            pred = CLASS_NAMES[attr][int(out0[attr].argmax(-1).item())]
            axes[r, c].imshow(img)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.5)
            axes[r, c].set_title(f"{attr}  →  {pred}", fontsize=15, fontweight="bold")
            axes[r, c].axis("off")
        divs.append(head_div(cams))

    fig.suptitle("Level 4 — ResNet-18 Grad-CAM (last conv, per-head saliency)",
                 fontsize=17, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "level4_gradcam_cnn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    p = Path("tables/level4_cam_diff.json")
    d = json.loads(p.read_text()) if p.exists() else {}
    e = d.get("cnn_resnet18", {})
    e["method"] = "standard Grad-CAM on layer4 (last conv)"
    e["figure_mean_3img"] = float(np.mean(divs))   # illustrative; report uses mean_20img
    e["figure_images"] = [list(t) for t in TARGETS]
    d["cnn_resnet18"] = e
    p.write_text(json.dumps(d, indent=2))
    print(f"wrote figures/level4_gradcam_cnn.png | CNN head-div mean={np.mean(divs):.3f} "
          f"per={[round(x,3) for x in divs]}")


if __name__ == "__main__":
    main()
