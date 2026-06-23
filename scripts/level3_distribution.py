"""Level 3 — Set A class-distribution figure for the PPT (imbalance at a glance).

Produces one figure with 3 horizontal-bar panels (weather / scene / timeofday),
each sorted descending, annotated with count + percentage. Built from the actual
Set A train split (the data Level 3 trains on). foggy=0 is drawn as an explicit
zero bar to make the global-zero limitation visible.

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level3_distribution.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.utils.metrics import CLASS_NAMES

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
TITLES = {"weather": "Weather (6)", "scene": "Scene (3)", "timeofday": "Time of Day (3)"}


def main():
    d = BDDAttrDataset("data/set_a", "train")
    N = len(d)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Set A train — class distribution  (N = {N:,})", fontsize=15, y=1.02)

    for ax, a in zip(axes, ATTRIBUTES):
        counts = d.class_counts(a).numpy().astype(int)
        names = CLASS_NAMES[a]
        order = np.argsort(counts)               # ascending -> longest bar on top
        counts, names = counts[order], [names[i] for i in order]
        pct = 100.0 * counts / N

        # color gradient: majority dark blue -> minority light; zero class red
        cmax = counts.max()
        colors = []
        for c in counts:
            if c == 0:
                colors.append("#d62728")          # zero-support -> red
            else:
                t = c / cmax
                colors.append((0.85 - 0.65 * t, 0.88 - 0.55 * t, 0.95 - 0.25 * t))

        y = np.arange(len(counts))
        ax.barh(y, counts, color=colors, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
        ax.set_title(TITLES[a], fontsize=13, fontweight="bold")
        ax.set_xlabel("images")
        ax.set_xlim(0, cmax * 1.18)
        ax.grid(axis="x", alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        for yi, (c, p) in enumerate(zip(counts, pct)):
            label = f"{c:,} ({p:.0f}%)" if c > 0 else "0  (none)"
            ax.text(c + cmax * 0.015, yi, label, va="center", fontsize=10,
                    color="#d62728" if c == 0 else "#222")

    fig.tight_layout()
    out = FIG_DIR / "level3_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
