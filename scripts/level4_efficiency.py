"""Level 4 — efficiency: Params / FLOPs / FPS vs Avg-MF1 Pareto front.

Params + FLOPs are hardware-independent (fine to compute on H100). FPS is
hardware-dependent: per the assignment it must be measured on **Colab T4** for
grading — the H100 number printed here is reference/sanity only. The notebook
re-runs ``measure_fps`` on T4 to fill the Pareto x-axis.

Avg-MF1 per backbone is read from the saved history/metrics:
    vgg16/resnet18/resnet50  <- checkpoints/level1_{name}_history.json (Level 1)
    vit_s16                  <- tables/level3_best_metrics.json (best Level 3),
                                fallback checkpoints/level2_vit_pretrained_history.json

Run (H100, reference FPS):
  CUDA_VISIBLE_DEVICES=0 /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level4_efficiency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models.resnet import resnet18, resnet50
from src.models.vgg import VGG16
from src.models.vit import vit_small_patch16_224
from src.utils.efficiency import count_flops, count_parameters, measure_fps

TBL = Path("tables")
FIG = Path("figures")
CKPT = Path("checkpoints")
FIG.mkdir(exist_ok=True)
TBL.mkdir(exist_ok=True)

MODELS = {
    "vgg16": VGG16,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "vit_s16": vit_small_patch16_224,
}


def avg_mf1_for(name: str) -> float | None:
    if name == "vit_s16":
        p = TBL / "level3_best_metrics.json"
        if p.exists():
            return json.loads(p.read_text())["best_val_avg_mf1"]
        p = CKPT / "level2_vit_pretrained_history.json"
        if p.exists():
            return max(json.loads(p.read_text())["val_avg_mf1"])
        return None
    p = CKPT / f"level1_{name}_history.json"
    if p.exists():
        return max(json.loads(p.read_text())["val_avg_mf1"])
    return None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = "H100 (reference — measure on T4 for grading)" if device.type == "cuda" else device.type
    print(f"device={device} | FPS hardware: {hw}\n")

    rows = []
    for name, fn in MODELS.items():
        model = fn().to(device)
        params = count_parameters(model)
        flops, _ = count_flops(model, device)
        fps = measure_fps(model, device, batch_size=1)
        mf1 = avg_mf1_for(name)
        rows.append({"name": name, "params_M": params / 1e6, "gflops": flops / 1e9,
                     "fps": fps, "avg_mf1": mf1})
        print(f"{name:10s} params={params/1e6:6.2f}M  FLOPs={flops/1e9:6.2f}G  "
              f"FPS={fps:7.1f}  Avg-MF1={mf1 if mf1 is None else round(mf1,4)}")
        del model
        torch.cuda.empty_cache()

    # markdown table
    lines = [
        "# Level 4 — Efficiency (backbone comparison)",
        "",
        f"Params/FLOPs hardware-independent. **FPS below = {hw}.**",
        "Avg-MF1 = best val (Level 1 for CNNs, best Level 3 for ViT).",
        "",
        "| backbone | Params (M) | FLOPs (G) | FPS | Avg-MF1 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mf1 = "—" if r["avg_mf1"] is None else f"{r['avg_mf1']:.4f}"
        lines.append(f"| {r['name']} | {r['params_M']:.2f} | {r['gflops']:.2f} | "
                     f"{r['fps']:.1f} | {mf1} |")
    (TBL / "level4_efficiency.md").write_text("\n".join(lines) + "\n")
    (TBL / "level4_efficiency.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote tables/level4_efficiency.md + .json")

    # Pareto front: FPS (x) vs Avg-MF1 (y)
    pts = [r for r in rows if r["avg_mf1"] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for r in pts:
            ax.scatter(r["fps"], r["avg_mf1"], s=80)
            ax.annotate(f"{r['name']}\n({r['params_M']:.1f}M, {r['gflops']:.1f}G)",
                        (r["fps"], r["avg_mf1"]), textcoords="offset points",
                        xytext=(8, 4), fontsize=8)
        ax.set_xlabel(f"FPS  ({'H100 — replace with T4' if device.type=='cuda' else device.type})")
        ax.set_ylabel("val Avg-MF1")
        ax.set_title("Level 4 — Efficiency vs Accuracy (Pareto)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "level4_pareto.png", dpi=150)
        plt.close(fig)
        print("wrote figures/level4_pareto.png")


if __name__ == "__main__":
    main()
