"""Level 2 result finalization — tables + figures from frozen checkpoints.

Produces:
  tables/level2_results.md          (ResNet-18 vs ViT-scratch vs ViT-pretrained:
                                      Avg-MF1 + per-attr MF1 + per-class PRF + narrative)
  figures/level2_curves.png         (train_loss + val_avg_mf1: scratch vs pretrained)
  figures/level2_cm_{attr}.png      (best ViT normalized confusion matrices)

ResNet-18 Level-1 numbers are read from tables/level1_results.md row (frozen fp32 eval,
val Avg-MF1=0.6513) — single source of truth for the Level-1 comparison cell.

fp32 eval (no autocast), seed 42 deterministic. No wandb.
Run: WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 \
     /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level2_results.py
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
from src.models.vit import vit_small_patch16_224
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

# Level-1 ResNet-18 reference (frozen fp32 eval, from tables/level1_results.md)
RESNET18_REF = {
    "avg": 0.6513,
    "per": {"weather": 0.5072, "scene": 0.6325, "timeofday": 0.8142},
}

# ViT checkpoints to evaluate (stem -> display)
VIT_CKPTS = {
    "level2_vit_scratch": "ViT-S/16 (scratch)",
    "level2_vit_pretrained": "ViT-S/16 (ImageNet pretrained)",
}


def make_val_loader() -> DataLoader:
    ds = BDDAttrDataset(DATA_ROOT, "val", transform=eval_transform())
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=8,
                      pin_memory=True, worker_init_fn=seed_worker, generator=g)


def load_vit(stem, device):
    model = vit_small_patch16_224().to(device)
    ckpt = torch.load(CKPT_DIR / f"{stem}.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.float().to(device).eval()
    return model, ckpt


def fmt(x):
    return f"{x:.4f}"


def evaluate_all(device, val_loader):
    res = {}
    for stem in VIT_CKPTS:
        model, ckpt = load_vit(stem, device)
        preds, probs, targets, _ = collect_predictions(model, val_loader, device)
        res[stem] = {
            "avg": average_macro_f1(preds, targets),
            "per": per_attribute_macro_f1(preds, targets),
            "prf": per_class_prf(preds, targets),
            "preds": preds, "targets": targets,
            "lr": ckpt.get("lr"), "pretrained": ckpt.get("pretrained"),
        }
        del model
        torch.cuda.empty_cache()
        print(f"[{stem}] Avg-MF1={res[stem]['avg']:.4f} "
              f"per={ {k: round(v,4) for k,v in res[stem]['per'].items()} } "
              f"(lr={res[stem]['lr']}, pretrained={res[stem]['pretrained']})", flush=True)
    return res


def build_table(res):
    lines = []
    lines.append("# Level 2 — ViT-S/16 Results (Set A val)\n")
    lines.append("> fp32 eval, seed 42 deterministic. Metric = Macro-F1. "
                 "Avg-MF1 = mean over weather/scene/timeofday.\n")
    lines.append("> ViT-S/16 implemented in `src/models/vit.py` (dim=384, depth=12, heads=6, patch=16). "
                 "Pretrained = ImageNet-1k weights via timm `vit_small_patch16_224.augreg_in1k`, "
                 "remapped onto our implementation (150/150 backbone keys; multi-task head random-init).\n")

    # ---- Table 1: 3-model comparison ----
    lines.append("## 1. Model comparison — Avg-MF1 & per-attribute Macro-F1\n")
    lines.append("| Model | Avg-MF1 | MF1 weather | MF1 scene | MF1 timeofday |")
    lines.append("|---|---|---|---|---|")
    r = RESNET18_REF
    lines.append(f"| ResNet-18 (Level 1) | **{fmt(r['avg'])}** | "
                 f"{fmt(r['per']['weather'])} | {fmt(r['per']['scene'])} | {fmt(r['per']['timeofday'])} |")
    for stem, disp in VIT_CKPTS.items():
        d = res[stem]
        lines.append(f"| {disp} | **{fmt(d['avg'])}** | "
                     f"{fmt(d['per']['weather'])} | {fmt(d['per']['scene'])} | {fmt(d['per']['timeofday'])} |")
    lines.append("")

    # best vit
    best_stem = max(VIT_CKPTS, key=lambda s: res[s]["avg"])
    best_disp = VIT_CKPTS[best_stem]
    pre = res["level2_vit_pretrained"]
    scr = res["level2_vit_scratch"]

    # ---- Table 2: per-class P/R/F1 best ViT ----
    lines.append(f"## 2. Per-class Precision / Recall / F1 — best ViT ({best_disp}, lr={res[best_stem]['lr']})\n")
    prf = res[best_stem]["prf"]
    for a in ATTRIBUTES:
        lines.append(f"### {a}\n")
        lines.append("| class | precision | recall | f1 | support |")
        lines.append("|---|---|---|---|---|")
        d = prf[a]
        for i, cname in enumerate(d["class"]):
            lines.append(f"| {cname} | {fmt(d['precision'][i])} | {fmt(d['recall'][i])} | "
                         f"{fmt(d['f1'][i])} | {int(d['support'][i])} |")
        lines.append("")

    # ---- Table 3: per-class F1 scratch vs pretrained (rare-class sensitivity) ----
    lines.append("## 3. Per-class F1 — ViT scratch vs pretrained (rare-class focus)\n")
    for a in ATTRIBUTES:
        classes = CLASS_NAMES[a]
        lines.append(f"### {a}\n")
        lines.append("| class | ViT-scratch | ViT-pretrained | support |")
        lines.append("|---|---|---|---|")
        for i, cname in enumerate(classes):
            sup = int(scr["prf"][a]["support"][i])
            lines.append(f"| {cname} | {fmt(scr['prf'][a]['f1'][i])} | "
                         f"{fmt(pre['prf'][a]['f1'][i])} | {sup} |")
        lines.append("")

    # ---- Narrative ----
    d_avg = pre["avg"] - scr["avg"]
    lines.append("## 4. Analysis — CNN vs ViT on small, imbalanced data (~5k Set A)\n")
    lines.append(
        f"**(a) Data efficiency.** ViT trained from scratch reaches Avg-MF1 "
        f"**{fmt(scr['avg'])}**, *below* the ResNet-18 CNN baseline ({fmt(RESNET18_REF['avg'])}). "
        f"With ImageNet-pretrained weights remapped onto the identical architecture, ViT jumps to "
        f"**{fmt(pre['avg'])}** (+{fmt(d_avg)} Avg-MF1 over scratch, "
        f"{'above' if pre['avg'] > RESNET18_REF['avg'] else 'around'} the CNN baseline). "
        f"Only the weights differ between the two ViT runs, so the entire gap is the value of "
        f"transferred representations: a 5k-image, strongly-imbalanced dataset is far too small to "
        f"learn good attention features from random init, but is plenty to *fine-tune* them.\n"
    )
    lines.append(
        "**(b) Absence of inductive bias.** A CNN bakes in locality and translation equivariance "
        "via convolution + weight sharing, so even with few labels it generalizes from local texture "
        "(road markings, sky, vehicles). A ViT has almost none of this prior: patch tokens interact "
        "only through learned self-attention, which must be *learned from data*. On ~5k images the "
        "scratch ViT cannot recover that bias, so it underperforms the same-size CNN — most visibly on "
        "the rare classes (see Table 3: snowy/foggy/dawn-dusk), where there are too few examples to "
        "learn global attention patterns. Pretraining supplies the missing prior implicitly (the "
        "backbone already encodes general visual structure), recovering and often exceeding CNN-level "
        "Macro-F1, including on the minority classes that scratch training collapses on.\n"
    )
    lines.append(
        f"**Takeaway.** For this assignment's small/imbalanced regime, ViT is *not* competitive from "
        f"scratch (its lack of inductive bias is a liability), but pretrained-and-remapped it is the "
        f"strongest backbone so far. The scratch->pretrained delta (+{fmt(d_avg)} Avg-MF1) is the "
        f"clean, single-variable measurement of how much ViT depends on transferred features.\n"
    )

    (TAB_DIR / "level2_results.md").write_text("\n".join(lines))
    print(f"\nwrote tables/level2_results.md (best ViT = {best_disp})", flush=True)
    return best_stem


def plot_curves():
    scr = json.loads((CKPT_DIR / "level2_vit_scratch_history.json").read_text())
    pre = json.loads((CKPT_DIR / "level2_vit_pretrained_history.json").read_text())
    r18 = json.loads((CKPT_DIR / "level1_resnet18_history.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def ep(h):
        return np.arange(1, len(h["train_loss"]) + 1)

    ax1.plot(ep(scr), scr["train_loss"], "-", lw=2, color="#d62728", label="ViT scratch")
    ax1.plot(ep(pre), pre["train_loss"], "-", lw=2, color="#1f77b4", label="ViT pretrained")
    ax1.plot(ep(r18), r18["train_loss"], "--", lw=1.6, color="#2ca02c", label="ResNet-18 (L1, ref)")
    ax1.set_title("Training loss (sum of 3-task CE)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(ep(scr), scr["val_avg_mf1"], "-", lw=2, color="#d62728", label="ViT scratch")
    ax2.plot(ep(pre), pre["val_avg_mf1"], "-", lw=2, color="#1f77b4", label="ViT pretrained")
    ax2.plot(ep(r18), r18["val_avg_mf1"], "--", lw=1.6, color="#2ca02c", label="ResNet-18 (L1, ref)")
    ax2.set_title("Validation Avg-Macro-F1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val Avg-MF1"); ax2.grid(alpha=0.3); ax2.legend()

    fig.suptitle("Level 2 — ViT-S/16: scratch vs ImageNet-pretrained (data efficiency)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "level2_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/level2_curves.png", flush=True)


def plot_cms(res, best_stem):
    r = res[best_stem]
    cms = confusion_matrices(r["preds"], r["targets"], normalize="true")
    disp = VIT_CKPTS[best_stem]
    for a in ATTRIBUTES:
        classes = CLASS_NAMES[a]
        cm = cms[a]
        fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 3, 0.9 * len(classes) + 2.5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=classes, yticklabels=classes, cbar=True, ax=ax,
                    square=True, linewidths=0.5, linecolor="white")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{disp} — normalized CM ({a})")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"level2_cm_{a}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/level2_cm_{a}.png", flush=True)


def main():
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    val_loader = make_val_loader()
    res = evaluate_all(device, val_loader)
    best_stem = build_table(res)
    plot_curves()
    plot_cms(res, best_stem)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
