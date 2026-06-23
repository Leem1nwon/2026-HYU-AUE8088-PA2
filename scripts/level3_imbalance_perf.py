"""Level 3 motivation — "imbalance -> minority classes underperform" evidence.

Baseline = plain ViT-pretrained with no imbalance handling (level3_baseline,
the starting point Level 3 improves on). For every class we pair its Set A train
frequency with the baseline's per-class val F1 / recall, then show that rarer
classes score lower.

Outputs:
  figures/level3_imbalance_perf.png   scatter: train frequency vs per-class F1
                                      (all 12 classes, trend line + Pearson r)
  tables/level3_imbalance_perf.md     per-class freq / support / recall / F1 table

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level3_imbalance_perf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = Path("tables"); TBL_DIR.mkdir(exist_ok=True)
BASE = "tables/level3_baseline_metrics.json"
ATTR_COLOR = {"weather": "#1f77b4", "scene": "#2ca02c", "timeofday": "#ff7f0e"}
ATTR_MARK = {"weather": "o", "scene": "s", "timeofday": "^"}


def main():
    m = json.loads(Path(BASE).read_text())
    prf = m["prf"]
    train = BDDAttrDataset("data/set_a", "train")
    N = len(train)

    rows = []  # (attr, cls, train_pct, val_support, recall, f1)
    for a in ATTRIBUTES:
        tc = train.class_counts(a).numpy().astype(int)
        p = prf[a]
        for i, cls in enumerate(p["class"]):
            rows.append((a, cls, 100.0 * tc[i] / N, p["support"][i],
                         p["recall"][i], p["f1"][i]))

    # ---- figure: train frequency (x) vs per-class F1 (y) ----
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    nz = [r for r in rows if r[2] > 0]               # exclude foggy (0 train) from fit
    xf = np.array([r[2] for r in nz]); yf = np.array([r[5] for r in nz])
    z = np.polyfit(xf, yf, 1); xs = np.linspace(0, max(xf) * 1.05, 50)
    r = np.corrcoef(xf, yf)[0, 1]
    ax.plot(xs, np.polyval(z, xs), "--", color="gray", lw=1.4,
            label=f"trend (Pearson r = {r:.2f}, n=11)")

    seen = set()
    for a, cls, pct, sup, rec, f1 in rows:
        lab = a if a not in seen else None; seen.add(a)
        if pct == 0:                                  # foggy: zero-support, drawn at origin
            ax.scatter(0, 0, marker="x", s=90, color="#d62728", zorder=5)
            ax.annotate("foggy (0 train)", (0, 0), textcoords="offset points",
                        xytext=(8, 6), fontsize=9, color="#d62728")
            continue
        ax.scatter(pct, f1, marker=ATTR_MARK[a], s=80, color=ATTR_COLOR[a],
                   edgecolor="white", zorder=4, label=lab)
        ax.annotate(cls, (pct, f1), textcoords="offset points", xytext=(6, -3),
                    fontsize=8.5, color="#333")

    ax.set_xlabel("Set A train frequency (%)")
    ax.set_ylabel("baseline per-class val F1")
    ax.set_title("Class imbalance -> minority classes underperform\n"
                 "(ViT-pretrained baseline, no imbalance handling)", fontsize=12)
    ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "level3_imbalance_perf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- table (sorted within attribute by train frequency, desc) ----
    lines = ["# Level 3 motivation — imbalance vs per-class performance",
             "",
             f"Baseline: **ViT-pretrained, plain CE (no imbalance handling)** — "
             f"Avg-MF1(final) = {m['final_val_avg_mf1']:.4f}. "
             f"Set A train N={N:,}; metrics on Set A val.",
             "",
             "| Attribute | Class | Train % | Val support | Recall | **F1** | Tier |",
             "|---|---|---:|---:|---:|---:|---|"]
    for a in ATTRIBUTES:
        sub = sorted([r for r in rows if r[0] == a], key=lambda r: -r[2])
        for j, (_, cls, pct, sup, rec, f1) in enumerate(sub):
            if pct == 0:
                tier = "zero-support"
            elif j == 0:
                tier = "majority"
            elif pct < 12:
                tier = "**minority**"
            else:
                tier = "mid"
            lines.append(f"| {a} | {cls} | {pct:.0f}% | {sup} | "
                         f"{rec:.3f} | {f1:.3f} | {tier} |")
    lines += ["",
              f"**Pearson r (train% vs F1, foggy excluded) = {r:.2f}** — "
              "rarer class -> lower F1. foggy (0 train) collapses to F1 = 0."]
    (TBL_DIR / "level3_imbalance_perf.md").write_text("\n".join(lines))

    # ---- console summary ----
    print("wrote figures/level3_imbalance_perf.png + tables/level3_imbalance_perf.md")
    print(f"Pearson r(train%, F1) = {r:.3f} (n=11, foggy excluded)")
    for a in ATTRIBUTES:
        sub = sorted([rr for rr in rows if rr[0] == a], key=lambda rr: -rr[2])
        maj, mino = sub[0], sub[-1]
        print(f"  {a:9s} majority {maj[1]:<12s} F1={maj[5]:.3f} | "
              f"rarest {mino[1]:<12s} F1={mino[5]:.3f}")


if __name__ == "__main__":
    main()
