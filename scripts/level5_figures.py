"""Level 5 — curation figures: picks distribution + DI/ablation bars.

figures/level5_picks_distribution.png  picks-1000 vs random-1000 vs Set A train,
                                       per attribute (% of set) — shows balancing.
figures/level5_di_ablation.png         (a) final Avg-MF1 per config with random
                                       reference; (b) DI% for 250/500/1000.

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level5_figures.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.utils.metrics import CLASS_NAMES

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
TBL = Path("tables")


def pct_dist(picks, a, n_classes):
    c = Counter(p[a] for p in picks)
    tot = max(len(picks), 1)
    return np.array([100.0 * c.get(k, 0) / tot for k in range(n_classes)])


def picks_distribution():
    picks = json.loads(Path("level5_picks.json").read_text())["picks"]
    rand = json.loads((TBL / "level5_picks_random.json").read_text())["picks"]
    train = BDDAttrDataset("data/set_a", "train")
    N = len(train)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, a in zip(axes, ATTRIBUTES):
        nc = len(CLASS_NAMES[a]); names = CLASS_NAMES[a]
        p = pct_dist(picks, a, nc)
        r = pct_dist(rand, a, nc)
        t = 100.0 * train.class_counts(a).numpy() / N
        x = np.arange(nc); w = 0.27
        ax.bar(x - w, t, w, label="Set A train", color="#bbbbbb")
        ax.bar(x, r, w, label="random-1000", color="#1f77b4")
        ax.bar(x + w, p, w, label="picks-1000", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_title(a, fontsize=12, fontweight="bold")
        ax.set_ylabel("% of set"); ax.grid(axis="y", alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Level 5 — picks rebalance the tail (picks vs random vs Set A train)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "level5_picks_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def di_ablation():
    M = {n: json.loads((TBL / f"level5_{n}_metrics.json").read_text())
         for n in ["setA_only", "random", "picks_250", "picks_500", "picks"]}
    fin = {n: M[n]["final_val_avg_mf1"] for n in M}
    rnd = fin["random"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) final Avg-MF1 per config
    order = ["setA_only", "random", "picks_250", "picks_500", "picks"]
    labels = ["Set A only", "random-1000", "picks-250", "picks-500", "picks-1000"]
    colors = ["#999999", "#1f77b4", "#ff7f0e", "#ff7f0e", "#d62728"]
    vals = [fin[n] for n in order]
    x = np.arange(len(order))
    ax1.bar(x, vals, color=colors)
    ax1.axhline(rnd, color="#1f77b4", ls="--", lw=1.3, label=f"random {rnd:.4f}")
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.0006, f"{v:.4f}", ha="center", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylim(0.69, 0.732); ax1.set_ylabel("val Avg-MF1 (final)")
    ax1.set_title("(a) Retrain result — picks-1000 best"); ax1.legend(fontsize=9)

    # (b) DI%
    di_order = ["picks_250", "picks_500", "picks"]
    di_labels = ["picks-250", "picks-500", "picks-1000"]
    dis = [(fin[n] - rnd) / rnd * 100 for n in di_order]
    xb = np.arange(len(di_order))
    bars = ax2.bar(xb, dis, color=["#ff7f0e", "#ff7f0e", "#d62728"])
    ax2.axhline(0, color="#333", lw=1)
    for xi, d in zip(xb, dis):
        ax2.text(xi, d + (0.05 if d >= 0 else -0.12), f"{d:+.2f}%", ha="center", fontsize=10,
                 fontweight="bold")
    ax2.set_xticks(xb); ax2.set_xticklabels(di_labels, fontsize=10)
    ax2.set_ylabel("DI vs random-1000 (%)")
    ax2.set_title("(b) DI — non-monotonic (1000 > 250 > 500)")
    ax2.set_ylim(min(dis) - 0.5, max(dis) + 0.6)

    fig.suptitle("Level 5 — Data Improvement vs random baseline (final-epoch)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "level5_di_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    picks_distribution()
    di_ablation()
    print("wrote figures/level5_picks_distribution.png, level5_di_ablation.png")
