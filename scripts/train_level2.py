"""Level 2 — train ViT-S/16 (our src/models/vit.py): scratch vs ImageNet-pretrained.

Three runs (sequential, single GPU):
  - level2-vit-scratch        : random init, lr 5e-4
  - level2-vit-pretrained     : remap ImageNet weights, lr 5e-4 (reference HP)
  - level2-vit-pretrained-lr1e4: remap ImageNet weights, lr 1e-4 (gentler fine-tune)

We try two fine-tune LRs because 5e-4 (the reference HP) can be too large for a
pretrained backbone and collapse it; we keep whichever pretrained run gives the
higher val Avg-MF1 as ``checkpoints/level2_vit_pretrained.pth`` and report both.

Common HP: AdamW wd=5e-2, CosineAnnealing, epochs=25, batch=64, AMP fp16, seed 42.
Checkpoints saved fp32 (T4-load compatible).

wandb: own account only (WANDB_API_KEY injected). If account check fails the caller
should run with WANDB_DISABLED=true (history.json logging still happens).

Run:
  WANDB_API_KEY=$(cat ~/.wandb_key) CUDA_VISIBLE_DEVICES=0 \
    /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/train_level2.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.vit_load_pretrained import load_pretrained_vit
from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import CLASS_NAMES, collect_predictions, confusion_matrices
from src.utils.seed import seed_worker, set_seed
from src.utils.trainer import MultiTaskTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform
from src.utils.wandb_logger import WandbLogger

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 64
EPOCHS = 25
WD = 5e-2
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

WANDB_PROJECT = "aue8088-pa2"
WANDB_TAGS = ["level2", "vit"]

# (run_name, pretrained?, lr, output stem)
RUNS = [
    ("level2-vit-scratch", False, 5e-4, "level2_vit_scratch"),
    ("level2-vit-pretrained", True, 5e-4, "level2_vit_pretrained_lr5e4"),
    ("level2-vit-pretrained-lr1e4", True, 1e-4, "level2_vit_pretrained_lr1e4"),
]


def make_loader(split: str, transform, shuffle: bool) -> DataLoader:
    ds = BDDAttrDataset(DATA_ROOT, split, transform=transform)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds, batch_size=BATCH, shuffle=shuffle, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )


def train_one(run_name, pretrained, lr, stem, device, train_loader, val_loader):
    set_seed(SEED, deterministic=True)
    model = vit_small_patch16_224().to(device)
    if pretrained:
        rep = load_pretrained_vit(model, verbose=False)
        print(f"  [pretrained] loaded {rep['n_loaded']} backbone keys "
              f"(missing={len(rep['missing'])} head params)", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    losses = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    cfg = TrainConfig(epochs=EPOCHS, lr=lr, weight_decay=WD, amp=True)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        run_name=run_name,
        config={
            "backbone": "vit_small_patch16_224", "pretrained": pretrained,
            "epochs": EPOCHS, "batch": BATCH, "lr": lr, "weight_decay": WD,
            "seed": SEED, "amp": "fp16", "loss_weights": cfg.loss_weights,
        },
        tags=WANDB_TAGS + (["pretrained"] if pretrained else ["scratch"]),
    )
    trainer = MultiTaskTrainer(model, optim, sched, losses, device, cfg, logger=logger)
    history = trainer.fit(train_loader, val_loader)

    val_pred, _, val_tgt, _ = collect_predictions(model, val_loader, device)
    cms = confusion_matrices(val_pred, val_tgt)
    for a in ATTRIBUTES:
        logger.log_confusion_matrix(f"final/cm_{a}", cms[a], CLASS_NAMES[a])
    logger.finish()

    torch.save(
        {"state_dict": model.float().state_dict(), "history": history,
         "seed": SEED, "pretrained": pretrained, "lr": lr},
        CKPT_DIR / f"{stem}.pth",
    )
    (CKPT_DIR / f"{stem}_history.json").write_text(json.dumps(history, indent=2))
    best = max(history["val_avg_mf1"])
    print(f"[{run_name}] DONE  best val Avg-MF1={best:.4f}\n", flush=True)
    return best


def main() -> None:
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | epochs={EPOCHS} batch={BATCH} wd={WD} | project={WANDB_PROJECT}", flush=True)

    train_loader = make_loader("train", train_transform(), shuffle=True)
    val_loader = make_loader("val", eval_transform(), shuffle=False)

    results = {}
    for run_name, pretrained, lr, stem in RUNS:
        print(f"\n===== {run_name} (pretrained={pretrained}, lr={lr}) =====", flush=True)
        results[stem] = train_one(run_name, pretrained, lr, stem, device, train_loader, val_loader)

    # pick best pretrained LR -> canonical level2_vit_pretrained.pth
    pre_stems = [s for (_, p, _, s) in RUNS if p]
    best_pre = max(pre_stems, key=lambda s: results[s])
    shutil.copyfile(CKPT_DIR / f"{best_pre}.pth", CKPT_DIR / "level2_vit_pretrained.pth")
    shutil.copyfile(CKPT_DIR / f"{best_pre}_history.json", CKPT_DIR / "level2_vit_pretrained_history.json")

    print("=" * 56)
    for stem, best in results.items():
        print(f"{stem:30s} best val Avg-MF1={best:.4f}")
    print(f"\nbest pretrained = {best_pre}  -> copied to level2_vit_pretrained.pth", flush=True)


if __name__ == "__main__":
    main()
