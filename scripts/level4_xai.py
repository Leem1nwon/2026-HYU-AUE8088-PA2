"""Level 4 — XAI: multi-task Grad-CAM (CNN) + ViT Grad-CAM + Attention Rollout.

For a few representative val images, show where each of the 3 attribute heads
looks (shared backbone, head-specific backprop):
  - CNN  (ResNet-18, level1 ckpt): standard Grad-CAM on layer4.
  - ViT  (best Level-3 ckpt):       token Grad-CAM on the last block, + rollout.

The CNN-vs-ViT comparison (localized conv saliency vs. attention spread) is the
"depth of interpretation" the assignment grades.

Outputs:
  figures/level4_gradcam_cnn.png    rows=images, cols=[orig, weather, scene, timeofday]
  figures/level4_gradcam_vit.png    rows=images, cols=[orig, weather, scene, timeofday, rollout]

Run (after Level 3 finishes):
  CUDA_VISIBLE_DEVICES=0 /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level4_xai.py
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
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import CLASS_NAMES
from src.utils.seed import set_seed
from src.utils.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.xai.gradcam import GradCAM, ViTGradCAM, attention_rollout

FIG = Path("figures")
CKPT = Path("checkpoints")
FIG.mkdir(exist_ok=True)

# weather classes to showcase (diverse lighting/texture); foggy absent in train
SHOWCASE_WEATHER = ["clear", "rainy", "snowy", "overcast"]
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denorm(x: torch.Tensor):
    img = (x.cpu() * STD + MEAN).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def pick_images(ds, device):
    """One val image per showcase weather class (fallback: first images)."""
    want = {WEATHER_CLASSES.index(c) for c in SHOWCASE_WEATHER}
    chosen, seen = [], set()
    for i in range(len(ds)):
        s = ds.samples[i]
        if s.weather in want and s.weather not in seen:
            seen.add(s.weather)
            chosen.append(i)
        if len(seen) == len(want):
            break
    if not chosen:
        chosen = list(range(min(4, len(ds))))
    return chosen


def pred_label(out, attr):
    idx = int(out[attr].argmax(dim=-1).item())
    return CLASS_NAMES[attr][idx]


def score_fn_for(attr):
    return lambda out: out[attr].max(dim=-1).values.sum()


def _norm01(c):
    c = np.asarray(c, dtype=np.float64)
    return (c - c.min()) / (c.max() - c.min() + 1e-8)


def head_divergence(cams):
    """Mean pairwise mean-absolute-difference between the 3 head CAMs (min-max
    normalized to [0,1]). Higher => the heads attend to more different regions."""
    n = [_norm01(c) for c in cams]
    pairs = [(0, 1), (0, 2), (1, 2)]
    return float(np.mean([np.abs(n[i] - n[j]).mean() for i, j in pairs]))


def load_model(fn, ckpt_path, device):
    model = fn().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def run_cnn(idxs, ds, device):
    path = CKPT / "level1_resnet18.pth"
    model = load_model(resnet18, path, device)
    cam_fn = GradCAM(model, model.layer4[-1])

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.7 * n))
    axes = axes.reshape(n, 4)
    diffs = []
    for r, idx in enumerate(idxs):
        x = ds[idx]["image"].unsqueeze(0).to(device)
        img = denorm(x[0])
        with torch.no_grad():
            out0 = model(x)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"input\nW:{pred_label(out0,'weather')}", fontsize=8)
        axes[r, 0].axis("off")
        cams = []
        for c, attr in enumerate(ATTRIBUTES, start=1):
            cam = cam_fn(x, score_fn_for(attr))[0].cpu().numpy()
            cams.append(cam)
            axes[r, c].imshow(img)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
            axes[r, c].set_title(f"{attr}\n->{pred_label(out0,attr)}", fontsize=8)
            axes[r, c].axis("off")
        diffs.append({"weather": WEATHER_CLASSES[ds.samples[idx].weather],
                      "head_divergence": head_divergence(cams)})
    fig.suptitle("Level 4 — ResNet-18 Grad-CAM (per-head saliency)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "level4_gradcam_cnn.png", dpi=150)
    plt.close(fig)
    print("wrote figures/level4_gradcam_cnn.png")
    del model
    torch.cuda.empty_cache()
    return diffs


def run_vit(idxs, ds, device):
    path = CKPT / "level3_best.pth"
    if not path.exists():
        path = CKPT / "level2_vit_pretrained.pth"
    model = load_model(vit_small_patch16_224, path, device)
    # blocks[-1] gives all-zero patch gradients in ViT; blocks[-4] is informative
    cam_fn = ViTGradCAM(model, model.blocks[-4])

    n = len(idxs)
    fig, axes = plt.subplots(n, 5, figsize=(13.5, 2.7 * n))
    axes = axes.reshape(n, 5)
    diffs = []
    for r, idx in enumerate(idxs):
        x = ds[idx]["image"].unsqueeze(0).to(device)
        img = denorm(x[0])
        with torch.no_grad():
            out0 = model(x)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"input\nW:{pred_label(out0,'weather')}", fontsize=8)
        axes[r, 0].axis("off")
        cams = []
        for c, attr in enumerate(ATTRIBUTES, start=1):
            cam = cam_fn(x, score_fn_for(attr))[0].cpu().numpy()
            cams.append(cam)
            axes[r, c].imshow(img)
            axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
            axes[r, c].set_title(f"{attr}\n->{pred_label(out0,attr)}", fontsize=8)
            axes[r, c].axis("off")
        diffs.append({"weather": WEATHER_CLASSES[ds.samples[idx].weather],
                      "head_divergence": head_divergence(cams)})
        roll = attention_rollout(model, x)[0].cpu().numpy()
        axes[r, 4].imshow(img)
        axes[r, 4].imshow(roll, cmap="jet", alpha=0.45)
        axes[r, 4].set_title("attn rollout\n(backbone)", fontsize=8)
        axes[r, 4].axis("off")
    fig.suptitle(f"Level 4 — ViT Grad-CAM + Attention Rollout  ({path.name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "level4_gradcam_vit.png", dpi=150)
    plt.close(fig)
    print("wrote figures/level4_gradcam_vit.png")
    del model
    torch.cuda.empty_cache()
    return diffs


def main() -> None:
    set_seed(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BDDAttrDataset("data/set_a", "val", transform=eval_transform())
    idxs = pick_images(ds, device)
    print(f"showcase val images: {idxs} "
          f"(weather={[WEATHER_CLASSES[ds.samples[i].weather] for i in idxs]})")
    cnn_diffs = run_cnn(idxs, ds, device)
    vit_diffs = run_vit(idxs, ds, device)

    # head-divergence: how differently the 3 heads attend (reproducible numbers)
    def summ(diffs):
        v = [d["head_divergence"] for d in diffs]
        return {"per_image": diffs, "mean": float(np.mean(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}
    out = {"metric": "mean pairwise mean-abs-diff of min-max-normalized head CAMs "
                      "(higher = heads attend to more different regions)",
           "cnn_resnet18": summ(cnn_diffs), "vit_level3_best": summ(vit_diffs)}
    Path("tables/level4_cam_diff.json").write_text(json.dumps(out, indent=2))
    c, v = out["cnn_resnet18"], out["vit_level3_best"]
    print(f"\nhead-divergence  CNN: {c['min']:.3f}-{c['max']:.3f} (mean {c['mean']:.3f}) | "
          f"ViT: {v['min']:.3f}-{v['max']:.3f} (mean {v['mean']:.3f})")
    print("wrote tables/level4_cam_diff.json")


if __name__ == "__main__":
    main()
