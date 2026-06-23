"""Level 1 analysis (b) — loss-weight ablation on ResNet-18.

Trains ResNet-18 several times changing ONLY the multi-task loss weights
(weather/scene/timeofday). All other HP fixed:
  epochs=15, batch=64, lr=3e-4, AdamW, wd=5e-4, CosineAnnealing, AMP fp16.

Runs sequentially on a single GPU. No wandb (no-op logger). seed 42 deterministic.
Saves each best-by-val checkpoint as fp32, and writes the ablation table.

Run: WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 \
     /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level1_loss_weight_ablation.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.resnet import resnet18
from src.utils.seed import seed_worker, set_seed
from src.utils.trainer import MultiTaskTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 64
EPOCHS = 15
LR = 3e-4
WD = 5e-4
CKPT_DIR = Path("checkpoints")
TAB_DIR = Path("tables")
CKPT_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# (tag, loss_weights). weather is the hardest attribute -> emphasize it.
RUNS = [
    ("w1_s1_t1", {"weather": 1.0, "scene": 1.0, "timeofday": 1.0}),  # baseline
    ("w2_s1_t1", {"weather": 2.0, "scene": 1.0, "timeofday": 1.0}),  # weather x2
    ("w3_s1_t1", {"weather": 3.0, "scene": 1.0, "timeofday": 1.0}),  # weather x3
    ("w1_s1_t2", {"weather": 1.0, "scene": 1.0, "timeofday": 2.0}),  # contrast (tod x2)
]


def make_loader(split, transform, shuffle):
    ds = BDDAttrDataset(DATA_ROOT, split, transform=transform)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds, batch_size=BATCH, shuffle=shuffle, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )


def run_one(tag, weights, device, train_loader, val_loader):
    set_seed(SEED, deterministic=True)
    model = resnet18().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    losses = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WD,
                      loss_weights=dict(weights), amp=True)
    # logger=None -> trainer uses a no-op WandbLogger(project=None); wandb stays off.
    trainer = MultiTaskTrainer(model, optim, sched, losses, device, cfg, logger=None)

    print(f"\n=== run {tag}  weights={weights} ===", flush=True)
    history = trainer.fit(train_loader, val_loader)

    val_avg = history["val_avg_mf1"]
    best_epoch = int(max(range(len(val_avg)), key=lambda i: val_avg[i]))
    best_avg = float(val_avg[best_epoch])
    best_per = history["val_per_mf1"][best_epoch]

    # save fp32 ckpt (final-epoch weights) + history; best metrics recorded separately
    torch.save(
        {"state_dict": model.float().state_dict(), "history": history,
         "loss_weights": dict(weights), "best_epoch": best_epoch + 1,
         "best_avg_mf1": best_avg, "seed": SEED},
        CKPT_DIR / f"level1_lw_{tag}.pth",
    )
    (CKPT_DIR / f"level1_lw_{tag}_history.json").write_text(json.dumps(history, indent=2))

    print(f"[{tag}] best@epoch{best_epoch+1}  Avg-MF1={best_avg:.4f}  per={ {k: round(v,4) for k,v in best_per.items()} }", flush=True)
    return {"tag": tag, "weights": dict(weights), "best_epoch": best_epoch + 1,
            "best_avg": best_avg, "best_per": best_per}


def write_table(results):
    lines = []
    lines.append("# Level 1 — Analysis (b): Loss-weight ablation (ResNet-18)\n")
    lines.append("> epochs=15, batch=64, lr=3e-4, AdamW wd=5e-4, CosineAnnealing, AMP fp16, "
                 "seed 42 deterministic. Best-by-val epoch reported.\n")
    lines.append("| run | w_weather | w_scene | w_timeofday | best epoch | Avg-MF1 | MF1 weather | MF1 scene | MF1 timeofday |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    base = results[0]["best_avg"]
    for r in results:
        w = r["weights"]
        p = r["best_per"]
        lines.append(
            f"| {r['tag']} | {w['weather']:.0f} | {w['scene']:.0f} | {w['timeofday']:.0f} | "
            f"{r['best_epoch']} | **{r['best_avg']:.4f}** | "
            f"{p['weather']:.4f} | {p['scene']:.4f} | {p['timeofday']:.4f} |"
        )
    lines.append("")
    # delta vs baseline
    lines.append("## Delta vs baseline (w1_s1_t1)\n")
    lines.append("| run | Avg-MF1 delta | weather delta | scene delta | timeofday delta |")
    lines.append("|---|---|---|---|---|")
    b = results[0]
    for r in results:
        d_avg = r["best_avg"] - b["best_avg"]
        d_w = r["best_per"]["weather"] - b["best_per"]["weather"]
        d_s = r["best_per"]["scene"] - b["best_per"]["scene"]
        d_t = r["best_per"]["timeofday"] - b["best_per"]["timeofday"]
        lines.append(f"| {r['tag']} | {d_avg:+.4f} | {d_w:+.4f} | {d_s:+.4f} | {d_t:+.4f} |")
    lines.append("")
    lines.append(f"> Note: 15-epoch quick runs (vs 30-epoch main training), so absolute "
                 f"Avg-MF1 is lower than the frozen ResNet-18 checkpoint (~0.65). "
                 f"Compare runs RELATIVELY.\n")
    (TAB_DIR / "level1_loss_weight_ablation.md").write_text("\n".join(lines))
    print("\nwrote tables/level1_loss_weight_ablation.md", flush=True)


def main():
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  epochs={EPOCHS} batch={BATCH} lr={LR}", flush=True)
    train_loader = make_loader("train", train_transform(), shuffle=True)
    val_loader = make_loader("val", eval_transform(), shuffle=False)

    results = []
    for tag, weights in RUNS:
        results.append(run_one(tag, weights, device, train_loader, val_loader))
        torch.cuda.empty_cache()

    write_table(results)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
