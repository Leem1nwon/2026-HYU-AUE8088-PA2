"""Level 3 — three figures, one per narrative beat (best-epoch basis).

beat1  figures/level3_beat1_ranking.png   Avg-MF1 ranking, bars colored by technique
                                          axis; baseline reference line.
beat2  figures/level3_beat2_maj_min.png   majority vs minority F1 per technique
                                          (lollipop): majority saturated ~0.92,
                                          minority gap persists, imbalance methods
                                          do not beat the baseline minority.
beat3  figures/level3_beat3_rareclass.png rare-class x technique heatmap; per-class
                                          winner boxed -> winners differ by class
                                          (cross-attribute trade-off).

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level3_3beat.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
RUNS = json.loads(Path("tables/level3_all_metrics.json").read_text())
BY = {r["stem"].replace("level3_", ""): r for r in RUNS}

AXIS = {
    "baseline": ("Baseline", "#7f7f7f"),
    "wce": ("Loss", "#1f77b4"), "focal": ("Loss", "#1f77b4"),
    "ldam": ("Loss", "#1f77b4"), "cb": ("Loss", "#1f77b4"),
    "sampler_weather": ("Sampling", "#2ca02c"), "sampler_joint": ("Sampling", "#2ca02c"),
    "randaug": ("Augmentation", "#ff7f0e"), "mixup_cutmix": ("Augmentation", "#ff7f0e"),
    "focal_sampler": ("Combined", "#9467bd"),
    "cb_sampler_randaug": ("Combined", "#9467bd"),
    "perattr_sampler_randaug": ("Combined", "#9467bd"),
}
LABEL = {"mixup_cutmix": "Mixup+CutMix", "randaug": "RandAugment", "baseline": "Baseline (plain CE)",
         "focal_sampler": "Focal + W-sampler", "ldam": "LDAM", "sampler_joint": "Sampler (W×T)",
         "wce": "Weighted CE", "focal": "Focal", "cb": "Class-Balanced",
         "perattr_sampler_randaug": "Per-attr + samp + randaug",
         "cb_sampler_randaug": "CB + samp + randaug", "sampler_weather": "Sampler (Weather)"}
MAJ = {"weather": [0], "scene": [0], "timeofday": [0, 1]}
MIN = {"weather": [1, 2, 3, 5], "scene": [2], "timeofday": [2]}


def avg(s): return BY[s]["best_val_avg_mf1"]
def maj_min(s):
    mj, mn = [], []
    for a in ["weather", "scene", "timeofday"]:
        f = BY[s]["prf"][a]["f1"]
        mj += [f[i] for i in MAJ[a]]; mn += [f[i] for i in MIN[a]]
    return np.mean(mj), np.mean(mn)


def beat1():
    order = sorted(BY, key=avg)                      # ascending -> best on top
    vals = [avg(s) for s in order]
    colors = [AXIS[s][1] for s in order]
    base = avg("baseline")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.axvline(base, color="#7f7f7f", ls="--", lw=1.4, label=f"baseline {base:.4f}")
    for yi, (s, v) in enumerate(zip(order, vals)):
        ax.text(v + 0.0008, yi, f"{v:.4f}", va="center", fontsize=8.5)
        if s == "mixup_cutmix":
            ax.text(v + 0.006, yi, "★ best", va="center", color="#d62728", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([LABEL[s] for s in order], fontsize=9)
    ax.set_xlim(0.69, 0.738); ax.set_xlabel("val Avg-MF1 (best epoch)")
    ax.set_title("Beat 1 — Augmentation tops; plain baseline beats most imbalance methods")
    seen = {}
    for s in order:
        ax_lbl, c = AXIS[s]
        if ax_lbl not in seen:
            seen[ax_lbl] = ax.barh(-1, 0, color=c, label=ax_lbl)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_ylim(-0.5, len(order) - 0.5)
    fig.tight_layout(); fig.savefig(FIG / "level3_beat1_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def beat2():
    order = sorted(BY, key=avg, reverse=True)        # best first (top)
    bm_maj, bm_min = maj_min("baseline")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(order))[::-1]
    for yi, s in zip(y, order):
        mj, mn = maj_min(s)
        ax.plot([mn, mj], [yi, yi], color="#cccccc", lw=2, zorder=1)
        ax.scatter(mj, yi, color="#1f77b4", s=55, zorder=3)
        ax.scatter(mn, yi, color="#d62728", s=55, zorder=3)
        ax.text(mj + 0.004, yi, f"{mj:.2f}", va="center", fontsize=7.5, color="#1f77b4")
        ax.text(mn - 0.004, yi, f"{mn:.2f}", va="center", ha="right", fontsize=7.5, color="#d62728")
    ax.axvline(bm_min, color="#d62728", ls=":", lw=1.3, label=f"baseline minority {bm_min:.3f}")
    ax.axvline(bm_maj, color="#1f77b4", ls=":", lw=1.3, label=f"baseline majority {bm_maj:.3f}")
    ax.scatter([], [], color="#1f77b4", label="majority classes")
    ax.scatter([], [], color="#d62728", label="minority classes")
    ax.set_yticks(y); ax.set_yticklabels([LABEL[s] for s in order], fontsize=9)
    ax.set_xlabel("mean per-class F1  (3 attributes combined)"); ax.set_xlim(0.55, 0.97)
    ax.set_title("Beat 2 — majority saturated (~0.92); minority gap persists,\n"
                 "imbalance methods do not lift minority above the baseline")
    ax.text(0.5, -0.13,
            "majority = clear · city street · daytime · night   |   "
            "minority = overcast · rainy · snowy · partly · residential · dawn/dusk   (foggy excluded)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
    ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "level3_beat2_maj_min.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def beat3():
    rare = [("weather", 3, "snowy"), ("weather", 2, "rainy"), ("weather", 5, "partly cloudy"),
            ("weather", 1, "overcast"), ("scene", 2, "residential"), ("timeofday", 2, "dawn/dusk")]
    order = sorted(BY, key=avg, reverse=True)
    M = np.array([[BY[s]["prf"][a]["f1"][ci] for s in order] for a, ci, _ in rare])
    fig, ax = plt.subplots(figsize=(13, 4.4))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0.45, vmax=0.85)
    for i in range(M.shape[0]):
        jmax = int(np.argmax(M[i]))
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.72 else "#222",
                    fontweight="bold" if j == jmax else "normal")
        ax.add_patch(Rectangle((jmax - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="red", lw=2.6, zorder=5))      # per-class winner
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[s] for s in order], rotation=40, ha="right", fontsize=8.5)
    base_j = order.index("baseline")
    ax.get_xticklabels()[base_j].set_color("#7f7f7f")
    ax.get_xticklabels()[base_j].set_fontweight("bold")
    ax.set_yticks(range(len(rare))); ax.set_yticklabels([f"{r[2]}" for r in rare], fontsize=9.5)
    ax.set_title("Beat 3 — per rare-class winner differs by technique (red box) "
                 "→ no single method wins all minority classes")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="per-class F1")
    fig.tight_layout(); fig.savefig(FIG / "level3_beat3_rareclass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # console: list winners
    print("Beat3 rare-class winners:")
    for i, (a, ci, nm) in enumerate(rare):
        j = int(np.argmax(M[i])); print(f"  {nm:14s}: {LABEL[order[j]]:24s} {M[i, j]:.3f} "
                                        f"(baseline {BY['baseline']['prf'][a]['f1'][ci]:.3f})")


if __name__ == "__main__":
    beat1(); beat2(); beat3()
    print("wrote figures/level3_beat1_ranking.png, level3_beat2_maj_min.png, level3_beat3_rareclass.png")
