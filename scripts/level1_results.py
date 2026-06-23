"""Level 1 result finalization — tables + figures from frozen checkpoints.

Produces:
  tables/level1_results.md            (Avg-MF1, per-attr MF1, per-class PRF, Top-1 / worst-class acc)
  figures/level1_loss_curves.png      (train_loss + val_avg_mf1, VGG dashed vs ResNet solid)
  figures/level1_cm_{weather,scene,timeofday}.png  (resnet18 normalized CM, seaborn)

No wandb. fp32 eval (no autocast). seed 42 deterministic.
Run: WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 \
     /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level1_results.py
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

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.resnet import resnet18, resnet50
from src.models.vgg import VGG16
from src.utils.metrics import (
    CLASS_NAMES,
    average_macro_f1,
    collect_predictions,
    confusion_matrices,
    per_attribute_macro_f1,
    per_class_prf,
)
from src.utils.seed import seed_worker, set_seed
from src.utils.transforms import eval_transform

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 128
CKPT_DIR = Path("checkpoints")
FIG_DIR = Path("figures")
TAB_DIR = Path("tables")
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

MODELS = {"resnet18": resnet18, "resnet50": resnet50, "vgg16": VGG16}
# pretty display names + skip-connection flag for the loss-curve figure
DISPLAY = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "vgg16": "VGG16"}
HAS_SKIP = {"resnet18": True, "resnet50": True, "vgg16": False}


def make_val_loader() -> DataLoader:
    ds = BDDAttrDataset(DATA_ROOT, "val", transform=eval_transform())
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds, batch_size=BATCH, shuffle=False, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )


def load_model(name, device):
    model = MODELS[name]().to(device)
    ckpt = torch.load(CKPT_DIR / f"level1_{name}.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.float().to(device).eval()
    return model


def top1_and_worst(preds, targets):
    """Per-attribute Top-1 accuracy + worst per-class recall (worst-class acc)."""
    out = {}
    for a in ATTRIBUTES:
        y_t, y_p = targets[a], preds[a]
        top1 = float((y_t == y_p).mean())
        # per-class recall (accuracy within each true class), ignore absent classes
        cms = confusion_matrices(preds, targets, normalize="true")
        diag = np.diag(cms[a])
        # only classes that actually appear in val targets
        present = np.array([np.any(y_t == c) for c in range(len(diag))])
        recalls = diag[present]
        worst = float(recalls.min()) if recalls.size else float("nan")
        worst_cls_idx = int(np.where(present)[0][int(np.argmin(recalls))]) if recalls.size else -1
        out[a] = {"top1": top1, "worst": worst, "worst_cls": worst_cls_idx}
    return out


def fmt(x):
    return f"{x:.4f}"


def build_results_table(device, val_loader):
    lines = []
    lines.append("# Level 1 — Validation Results (Set A val)\n")
    lines.append("> fp32 eval, seed 42 deterministic. Metric = Macro-F1. "
                 "Avg-MF1 = mean over weather/scene/timeofday.\n")

    # collect predictions for all 3 backbones
    all_results = {}
    for name in MODELS:
        model = load_model(name, device)
        preds, probs, targets, _ = collect_predictions(model, val_loader, device)
        avg = average_macro_f1(preds, targets)
        per = per_attribute_macro_f1(preds, targets)
        prf = per_class_prf(preds, targets)
        tw = top1_and_worst(preds, targets)
        all_results[name] = {
            "preds": preds, "targets": targets,
            "avg": avg, "per": per, "prf": prf, "tw": tw,
        }
        del model
        torch.cuda.empty_cache()
        print(f"[{name}] Avg-MF1={avg:.4f}  per={ {k: round(v,4) for k,v in per.items()} }", flush=True)

    # ---- Table 1: backbone comparison ----
    lines.append("## 1. Backbone comparison — Avg-MF1 & per-attribute Macro-F1\n")
    lines.append("| Backbone | Skip | Avg-MF1 | MF1 weather | MF1 scene | MF1 timeofday |")
    lines.append("|---|---|---|---|---|---|")
    for name in MODELS:
        r = all_results[name]
        skip = "yes" if HAS_SKIP[name] else "no"
        lines.append(
            f"| {DISPLAY[name]} | {skip} | **{fmt(r['avg'])}** | "
            f"{fmt(r['per']['weather'])} | {fmt(r['per']['scene'])} | {fmt(r['per']['timeofday'])} |"
        )
    lines.append("")

    # ---- Table 2: Top-1 & worst-class accuracy ----
    lines.append("## 2. Per-attribute Top-1 accuracy & Worst-class accuracy (recall)\n")
    lines.append("| Backbone | weather Top-1 | weather worst | scene Top-1 | scene worst | timeofday Top-1 | timeofday worst |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in MODELS:
        tw = all_results[name]["tw"]
        cells = []
        for a in ATTRIBUTES:
            wc = tw[a]["worst_cls"]
            wname = CLASS_NAMES[a][wc] if wc >= 0 else "-"
            cells.append(f"{fmt(tw[a]['top1'])}")
            cells.append(f"{fmt(tw[a]['worst'])} ({wname})")
        lines.append(f"| {DISPLAY[name]} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Table 3: per-class P/R/F1 per attribute (best model = resnet18) ----
    best_name = max(MODELS, key=lambda n: all_results[n]["avg"])
    lines.append(f"## 3. Per-class Precision / Recall / F1 — best backbone ({DISPLAY[best_name]})\n")
    prf = all_results[best_name]["prf"]
    for a in ATTRIBUTES:
        lines.append(f"### {a}\n")
        lines.append("| class | precision | recall | f1 | support |")
        lines.append("|---|---|---|---|---|")
        d = prf[a]
        for i, cname in enumerate(d["class"]):
            lines.append(
                f"| {cname} | {fmt(d['precision'][i])} | {fmt(d['recall'][i])} | "
                f"{fmt(d['f1'][i])} | {int(d['support'][i])} |"
            )
        lines.append("")

    # also dump full per-class PRF for ALL backbones (compact)
    lines.append("## 4. Per-class F1 for all backbones (per attribute)\n")
    for a in ATTRIBUTES:
        classes = CLASS_NAMES[a]
        header = "| class | " + " | ".join(DISPLAY[n] for n in MODELS) + " |"
        lines.append(f"### {a}\n")
        lines.append(header)
        lines.append("|" + "---|" * (len(MODELS) + 1))
        for i, cname in enumerate(classes):
            row = [cname]
            for n in MODELS:
                row.append(fmt(all_results[n]["prf"][a]["f1"][i]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    (TAB_DIR / "level1_results.md").write_text("\n".join(lines))
    print(f"\nwrote tables/level1_results.md  (best backbone = {best_name})", flush=True)
    return all_results, best_name


def plot_loss_curves():
    histories = {}
    for name in MODELS:
        h = json.loads((CKPT_DIR / f"level1_{name}_history.json").read_text())
        histories[name] = h

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"resnet18": "#1f77b4", "resnet50": "#2ca02c", "vgg16": "#d62728"}

    for name in MODELS:
        h = histories[name]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ls = "-" if HAS_SKIP[name] else "--"
        lw = 2.0
        label = f"{DISPLAY[name]} ({'skip' if HAS_SKIP[name] else 'no-skip'})"
        ax1.plot(epochs, h["train_loss"], ls=ls, lw=lw, color=colors[name], label=label)
        ax2.plot(epochs, h["val_avg_mf1"], ls=ls, lw=lw, color=colors[name], label=label)

    ax1.set_title("Training loss (sum of 3-task CE)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_title("Validation Avg-Macro-F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val Avg-MF1")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("Level 1 — Skip connection effect: VGG (no-skip, dashed) vs ResNet (skip, solid)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "level1_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/level1_loss_curves.png", flush=True)


def plot_confusion_matrices(all_results, best_name):
    r = all_results[best_name]
    cms = confusion_matrices(r["preds"], r["targets"], normalize="true")
    for a in ATTRIBUTES:
        classes = CLASS_NAMES[a]
        cm = cms[a]
        fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 3, 0.9 * len(classes) + 2.5))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes, cbar=True, ax=ax,
            square=True, linewidths=0.5, linecolor="white",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{DISPLAY[best_name]} — normalized CM ({a})")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"level1_cm_{a}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/level1_cm_{a}.png", flush=True)


def main():
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    val_loader = make_val_loader()
    all_results, best_name = build_results_table(device, val_loader)
    plot_loss_curves()
    plot_confusion_matrices(all_results, best_name)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
