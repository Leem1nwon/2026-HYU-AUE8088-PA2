"""Level 3 — PPT-ready comparison table (coherent, grouped, spelled-out).

Fixes the mixed best/final inconsistency in the old table: every row reports
Avg-MF1 AND per-attribute MF1 at the SAME epoch (the best-Avg-MF1 epoch), so the
three per-attribute columns average exactly to the Avg-MF1 column.

Rows are grouped by the README technique axis (Baseline / Loss / Sampling /
Augmentation / Combined), terms are spelled out, and a Delta-vs-baseline column
is added. A legend explains every abbreviation.

Output: tables/level3_results_clean.md

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level3_results_clean.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CKPT = Path("checkpoints")
TBL = Path("tables"); TBL.mkdir(exist_ok=True)

LOSS = {"plain": "—", "wce": "Weighted CE", "focal": "Focal", "ldam": "LDAM",
        "cb": "Class-Balanced", "perattr": "Per-attr†"}
SAMP = {None: "—", "weather": "Weather-bal.", "joint": "Joint (W×T)"}
AUG = {"basic": "—", "randaug": "RandAugment", "mix": "Mixup+CutMix"}

GROUPS = {
    "Baseline": ["baseline"],
    "Loss-level": ["wce", "focal", "ldam", "cb"],
    "Sampling-level": ["sampler_joint", "sampler_weather"],
    "Augmentation": ["randaug", "mixup_cutmix"],
    "Combined (2+ axes)": ["focal_sampler", "perattr_sampler_randaug", "cb_sampler_randaug"],
}


def best_epoch_metrics(stem):
    """Return (avg, per-attr dict) at the epoch maximizing Avg-MF1."""
    h = json.loads((CKPT / f"level3_{stem}_history.json").read_text())
    vp = h["val_per_mf1"]
    avg = [(e["weather"] + e["scene"] + e["timeofday"]) / 3 for e in vp]
    bi = int(np.argmax(avg))
    return avg[bi], vp[bi]


def main():
    meta = {r["stem"].replace("level3_", ""): r
            for r in json.loads((TBL / "level3_all_metrics.json").read_text())}

    base_avg, _ = best_epoch_metrics("baseline")
    best_stem = max(meta, key=lambda s: best_epoch_metrics(s)[0])

    L = ["# Level 3 — Imbalance & Augmentation: 12-run ablation",
         "",
         "ViT-S/16 (ImageNet-pretrained) backbone · AdamW lr 1e-4 / wd 5e-2 · 25 ep · seed 42 · Set A val.",
         "**모든 수치는 best-Avg-MF1 epoch 기준** (per-attribute 3개의 평균 = Avg-MF1, 검산 일치).",
         "",
         "| Technique | Loss | Sampler | Aug | Avg-MF1 | Δ base | weather | scene | timeofday |",
         "|---|---|---|---|---:|---:|---:|---:|---:|"]

    for g, stems in GROUPS.items():
        L.append(f"| **{g}** | | | | | | | | |")
        rows = []
        for s in stems:
            if s not in meta:
                continue
            avg, per = best_epoch_metrics(s)
            rows.append((s, avg, per))
        rows.sort(key=lambda x: -x[1])
        for s, avg, per in rows:
            r = meta[s]
            tag = " ⭐best" if s == best_stem else (" (ref)" if s == "baseline" else "")
            d = "ref" if s == "baseline" else f"{avg - base_avg:+.4f}"
            name = s.replace("_", " ")
            L.append(f"| {name}{tag} | {LOSS[r['loss']]} | {SAMP[r['sampler']]} | "
                     f"{AUG[r['aug']]} | {avg:.4f} | {d} | "
                     f"{per['weather']:.3f} | {per['scene']:.3f} | {per['timeofday']:.3f} |")

    L += ["",
          "### Legend",
          "- **Weighted CE** = 클래스 빈도 역가중 Cross-Entropy · **Focal** = (1−p)^γ down-weighting · "
          "**LDAM** = label-distribution-aware margin · **Class-Balanced** = effective-number 재가중.",
          "- **Sampler** — *Weather-bal.* = weather 빈도 역수 샘플링 · *Joint (W×T)* = weather×timeofday 결합 균형.",
          "- **Aug** — *RandAugment* = 자동 증강 정책 · *Mixup+CutMix* = 3-head 라벨 혼합 확장.",
          "- **†Per-attr** = 속성별 다른 loss (weather:LDAM / scene:Class-Balanced / timeofday:CE).",
          f"- **Δ base** = Avg-MF1 − baseline({base_avg:.4f} plain CE).",
          "",
          f"⭐ **best = {best_stem.replace('_',' ')}** (Avg-MF1 {best_epoch_metrics(best_stem)[0]:.4f}) "
          "→ level3_best.pth, Level 5 base 모델."]

    out = TBL / "level3_results_clean.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}\nbest = {best_stem} ({best_epoch_metrics(best_stem)[0]:.4f})")
    print("\n".join(L))


if __name__ == "__main__":
    main()
