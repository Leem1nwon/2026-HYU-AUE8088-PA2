"""Level 4 — per-attribute normalized confusion matrices with confused cells boxed.

Row-normalized CM (row = true class, diagonal = recall, off-diagonal = confusion
rate). Off-diagonal cells whose confusion rate is high get a RED border so the
"these two classes get confused" pairs pop out immediately. The worst-recall
diagonal cell per attribute gets an orange border.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/level4_confusion.py [ckpt] [thr]
  default ckpt = checkpoints/level3_best.pth, thr = 0.15

Output: figures/level4_confusion.png (1x3 panel, PPT-ready)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import CLASS_NAMES, collect_predictions, confusion_matrices
from src.utils.transforms import eval_transform

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/level3_best.pth"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15   # off-diag confusion to box
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
TITLES = {"weather": "Weather", "scene": "Scene", "timeofday": "Time of Day"}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_small_patch16_224().to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    model.float().to(device).eval()

    val = BDDAttrDataset("data/set_a", "val", transform=eval_transform())
    loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    preds, _, tgts, _ = collect_predictions(model, loader, device)
    cms = confusion_matrices(preds, tgts, normalize="true")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, a in zip(axes, ATTRIBUTES):
        cm = cms[a]
        names = CLASS_NAMES[a]
        n = len(names)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

        # support per true class (to flag zero-support rows like foggy)
        support = np.array([(tgts[a] == i).sum() for i in range(n)])
        diag = np.diag(cm)
        # worst-recall class among classes that actually appear
        present = np.where(support > 0)[0]
        worst = present[np.argmin(diag[present])] if len(present) else -1

        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                txt = f"{v:.2f}" if support[i] > 0 else "—"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 else "#222")
                # box: high off-diagonal confusion -> red
                if i != j and support[i] > 0 and v >= THR:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="red", lw=2.6, zorder=5))
        # worst-recall diagonal -> orange box
        if worst >= 0:
            ax.add_patch(Rectangle((worst - 0.5, worst - 0.5), 1, 1, fill=False,
                                   edgecolor="darkorange", lw=2.6, ls="--", zorder=5))

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{TITLES[a]}  (worst recall: {names[worst]} {diag[worst]:.2f})"
                     if worst >= 0 else TITLES[a], fontsize=11)

    # one shared legend
    fig.legend(handles=[
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", lw=2.6,
                  label=f"strong confusion (off-diag ≥ {THR:.2f})"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="darkorange", lw=2.6, ls="--",
                  label="worst-recall class"),
    ], loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Normalized confusion matrices — {Path(CKPT).stem} (Set A val)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIG / "level4_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # console: list the boxed confusion pairs
    print(f"wrote {out}  (ckpt={CKPT}, thr={THR})")
    for a in ATTRIBUTES:
        cm = cms[a]; names = CLASS_NAMES[a]
        pairs = [(names[i], names[j], cm[i, j]) for i in range(len(names))
                 for j in range(len(names)) if i != j and cm[i, j] >= THR]
        pairs.sort(key=lambda x: -x[2])
        print(f"  {a}: " + ", ".join(f"{t}->{p} {v:.2f}" for t, p, v in pairs))


if __name__ == "__main__":
    main()
