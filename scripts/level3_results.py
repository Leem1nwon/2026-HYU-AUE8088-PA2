"""Level 3 — aggregate ablation results into tables + figures.

Reads ``tables/level3_*_metrics.json`` (written by train_level3.py) and emits:

  tables/level3_results.md            comparison: Avg-MF1 + per-attr MF1, sorted
  tables/level3_weather_perclass.md   weather per-class F1 across experiments
  figures/level3_weather_f1_heatmap.png   experiments x weather-class F1 heatmap
  figures/level3_minority_vs_majority.png  weather minority/majority mean-F1 bars
  figures/level3_cm_{attr}.png        confusion matrices of the best run

The weather attribute is the analysis focus (foggy=0, clear 62%, snowy 200).
Minority = {overcast, rainy, snowy, partly cloudy}; majority = {clear};
foggy excluded (0 train samples — unlearnable in Level 3).

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level3_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, WEATHER_CLASSES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import CLASS_NAMES, collect_predictions, confusion_matrices
from src.utils.transforms import eval_transform

TBL = Path("tables")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
CKPT = Path("checkpoints")

WEATHER_MINORITY = ["overcast", "rainy", "snowy", "partly cloudy"]
WEATHER_MAJORITY = ["clear"]


def load_metrics() -> list[dict]:
    out, seen = [], set()
    for p in sorted(TBL.glob("level3_*_metrics.json")):
        if p.stem in ("level3_all_metrics", "level3_best_metrics"):
            continue
        m = json.loads(p.read_text())
        if m["stem"] in seen:
            continue
        seen.add(m["stem"])
        out.append(m)
    out.sort(key=lambda m: m["best_val_avg_mf1"], reverse=True)
    return out


def comparison_table(metrics: list[dict]) -> None:
    lines = [
        "# Level 3 — Imbalance & Augmentation (ViT-S/16, ImageNet init)",
        "",
        "HP: AdamW lr 1e-4 / wd 5e-2, CosineAnnealing, epochs 25, batch 64, AMP fp16, seed 42.",
        "`best` = max val Avg-MF1 over epochs; per-attr MF1 = final epoch.",
        "",
        "| run | loss | sampler | aug | **best Avg-MF1** | MF1 weather | MF1 scene | MF1 timeofday |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = next((m for m in metrics if m["name"] == "baseline"), None)
    for m in metrics:
        per = m["final_per_mf1"]
        flag = " ⭐" if m is metrics[0] else ""
        lines.append(
            f"| {m['name']}{flag} | {m['loss']} | {m['sampler'] or '—'} | {m['aug']} | "
            f"{m['best_val_avg_mf1']:.4f} | {per['weather']:.3f} | {per['scene']:.3f} | "
            f"{per['timeofday']:.3f} |"
        )
    if base is not None:
        lines += ["", f"baseline best Avg-MF1 = **{base['best_val_avg_mf1']:.4f}** (reference)."]
    (TBL / "level3_results.md").write_text("\n".join(lines) + "\n")
    print("wrote tables/level3_results.md")


def weather_f1_matrix(metrics: list[dict]) -> tuple[np.ndarray, list[str]]:
    """rows = experiments, cols = weather classes, value = per-class F1 (final)."""
    names = [m["name"] for m in metrics]
    mat = np.array([m["prf"]["weather"]["f1"] for m in metrics])  # (R, 6)
    return mat, names


def weather_perclass_table(metrics: list[dict]) -> None:
    mat, names = weather_f1_matrix(metrics)
    header = "| run | " + " | ".join(WEATHER_CLASSES) + " | min-mean | maj |"
    sep = "|" + "---|" * (len(WEATHER_CLASSES) + 3)
    lines = ["# Level 3 — weather per-class F1", "", "(foggy = 0 train samples → unlearnable)", "",
             header, sep]
    min_idx = [WEATHER_CLASSES.index(c) for c in WEATHER_MINORITY]
    maj_idx = [WEATHER_CLASSES.index(c) for c in WEATHER_MAJORITY]
    for name, row in zip(names, mat):
        cells = " | ".join(f"{v:.3f}" for v in row)
        lines.append(f"| {name} | {cells} | {row[min_idx].mean():.3f} | {row[maj_idx].mean():.3f} |")
    (TBL / "level3_weather_perclass.md").write_text("\n".join(lines) + "\n")
    print("wrote tables/level3_weather_perclass.md")


def heatmap_figure(metrics: list[dict]) -> None:
    mat, names = weather_f1_matrix(metrics)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=WEATHER_CLASSES, yticklabels=names, ax=ax, cbar_kws={"label": "F1"})
    ax.set_title("Level 3 — weather per-class F1 (rows sorted by Avg-MF1)")
    ax.set_xlabel("weather class"); ax.set_ylabel("experiment")
    fig.tight_layout()
    fig.savefig(FIG / "level3_weather_f1_heatmap.png", dpi=150)
    plt.close(fig)
    print("wrote figures/level3_weather_f1_heatmap.png")


def minority_majority_figure(metrics: list[dict]) -> None:
    mat, names = weather_f1_matrix(metrics)
    min_idx = [WEATHER_CLASSES.index(c) for c in WEATHER_MINORITY]
    maj_idx = [WEATHER_CLASSES.index(c) for c in WEATHER_MAJORITY]
    minority = mat[:, min_idx].mean(axis=1)
    majority = mat[:, maj_idx].mean(axis=1)

    order = np.argsort(minority)  # ascending so improvements read left->right
    names = [names[i] for i in order]
    minority, majority = minority[order], majority[order]

    x = np.arange(len(names)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(names)), 4.5))
    ax.bar(x - w / 2, minority, w, label="minority mean-F1 (overcast/rainy/snowy/partly)")
    ax.bar(x + w / 2, majority, w, label="majority F1 (clear)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("weather F1"); ax.set_ylim(0, 1)
    ax.set_title("Level 3 — weather minority vs majority F1 by technique")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "level3_minority_vs_majority.png", dpi=150)
    plt.close(fig)
    print("wrote figures/level3_minority_vs_majority.png")


def best_confusion_matrices() -> None:
    ckpt_path = CKPT / "level3_best.pth"
    if not ckpt_path.exists():
        print("level3_best.pth not found — skip CM figures.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_small_patch16_224().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    name = ckpt.get("spec", {}).get("name", "best")

    val_loader = DataLoader(
        BDDAttrDataset("data/set_a", "val", transform=eval_transform()),
        batch_size=64, shuffle=False, num_workers=8, pin_memory=True,
    )
    preds, _, tgts, _ = collect_predictions(model, val_loader, device)
    cms = confusion_matrices(preds, tgts)
    for a in ATTRIBUTES:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cms[a], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES[a], yticklabels=CLASS_NAMES[a], ax=ax)
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        ax.set_title(f"level3 best ({name}) — {a}")
        fig.tight_layout()
        fig.savefig(FIG / f"level3_cm_{a}.png", dpi=150)
        plt.close(fig)
    print(f"wrote figures/level3_cm_*.png (best run = {name})")


def main() -> None:
    metrics = load_metrics()
    if not metrics:
        print("no level3 metrics found in tables/. Run train_level3.py first.")
        return
    print(f"loaded {len(metrics)} experiment metrics")
    comparison_table(metrics)
    weather_perclass_table(metrics)
    heatmap_figure(metrics)
    minority_majority_figure(metrics)
    best_confusion_matrices()


if __name__ == "__main__":
    main()
