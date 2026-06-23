"""Level 5 step 3 — retrain (Set A + picks) for DI and ablation.

Notebook step 3, with the backbone = our best model (ViT) and the best recipe
(ImageNet-pretrained init, mixup-cutmix, AdamW lr 1e-4 / wd 5e-2, 25 epochs,
seed 42) so this is genuinely "best model retrained". All runs share the recipe,
so any val Avg-MF1 difference is due to the picks.

Runs:
  setA_only   Set A only (no picks)        reference
  random      + random-1000 picks          DI denominator
  picks       + my 1000 picks              submitted (DI numerator)
  picks_250   + my top-250                 ablation
  picks_500   + my top-500                 ablation

Resumable (skips runs whose metrics json exists). DataLoader hardened.

Run:
  WANDB_API_KEY=$(cat ~/.wandb_key) CUDA_VISIBLE_DEVICES=0 \
    /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level5_retrain.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.vit_load_pretrained import load_pretrained_vit
from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import (
    CLASS_NAMES, average_macro_f1, collect_predictions, confusion_matrices,
    per_attribute_macro_f1, per_class_prf,
)
from src.utils.seed import seed_worker, set_seed
from src.utils.trainer import MixupTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform
from src.utils.wandb_logger import WandbLogger

SEED = 42
DATA_ROOT = "data/set_a"
BATCH, NUM_WORKERS, LOADER_TIMEOUT = 64, 4, 180
EPOCHS, LR, WD = 25, 1e-4, 5e-2
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
TBL_DIR = Path("tables"); TBL_DIR.mkdir(exist_ok=True)
WANDB_PROJECT, WANDB_TAGS = "aue8088-pa2", ["level5"]

RUNS = [
    ("setA_only", None),
    ("random", "tables/level5_picks_random.json"),
    ("picks", "level5_picks.json"),
    ("picks_250", "tables/level5_picks_250.json"),
    ("picks_500", "tables/level5_picks_500.json"),
]


def load_extra(path):
    if path is None:
        return None
    picks = json.loads(Path(path).read_text())["picks"]
    return [(p["image_id"], p["weather"], p["scene"], p["timeofday"]) for p in picks]


def make_loader(ds, shuffle, g):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=True, worker_init_fn=seed_worker, generator=g,
                      timeout=LOADER_TIMEOUT, persistent_workers=True)


def train_one(name, picks_path, device, val_loader):
    set_seed(SEED, deterministic=True)
    extra = load_extra(picks_path)
    n_extra = 0 if extra is None else len(extra)
    train_ds = BDDAttrDataset(DATA_ROOT, "train", transform=train_transform(), extra_picks=extra)
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = make_loader(train_ds, True, g)

    model = vit_small_patch16_224().to(device)
    load_pretrained_vit(model, verbose=False)
    loss_fns = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WD, amp=True)

    logger = WandbLogger(
        project=WANDB_PROJECT, run_name=f"level5-{name}",
        config={"backbone": "vit_small_patch16_224", "pretrained": True, "aug": "mixup-cutmix",
                "picks": name, "n_extra": n_extra, "train_n": len(train_ds),
                "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "seed": SEED},
        tags=WANDB_TAGS + [name])
    trainer = MixupTrainer(model, optim, sched, loss_fns, device, cfg, logger=logger)
    history = trainer.fit(train_loader, val_loader)

    # final-epoch eval (model currently holds final weights) — used for the fair DI
    # comparison (best-epoch DI is confounded by the high-variance random baseline).
    val_pred, _, val_tgt, _ = collect_predictions(model, val_loader, device)
    for a in ATTRIBUTES:
        logger.log_confusion_matrix(f"final/cm_{a}", confusion_matrices(val_pred, val_tgt)[a], CLASS_NAMES[a])
    logger.finish()
    final_avg = average_macro_f1(val_pred, val_tgt)

    # deploy the BEST-Avg-MF1 epoch so the submitted checkpoint is the strongest
    # (consistent with Level 3) and reproduces best_val_avg_mf1 on eval.
    best_hist = trainer.best_val_avg_mf1
    torch.save({"state_dict": trainer.best_state, "history": history,
                "best_epoch": trainer.best_epoch,
                "seed": SEED, "picks": name, "n_extra": n_extra},
               CKPT_DIR / f"level5_{name}.pth")
    metrics = {"name": name, "n_extra": n_extra, "train_n": len(train_ds),
               "best_val_avg_mf1": best_hist, "best_epoch": trainer.best_epoch,
               "final_val_avg_mf1": final_avg,
               "final_per_mf1": per_attribute_macro_f1(val_pred, val_tgt),
               "prf": per_class_prf(val_pred, val_tgt)}
    (TBL_DIR / f"level5_{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{name}] DONE n_extra={n_extra} train_n={len(train_ds)} "
          f"best(ep{trainer.best_epoch})={best_hist:.4f} final={final_avg:.4f}", flush=True)

    del trainer, train_loader, train_ds, model, optim, sched
    gc.collect(); torch.cuda.empty_cache()
    return metrics


def main():
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | epochs={EPOCHS} lr={LR} | {len(RUNS)} runs", flush=True)
    val_loader = DataLoader(
        BDDAttrDataset(DATA_ROOT, "val", transform=eval_transform()),
        batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        timeout=LOADER_TIMEOUT, persistent_workers=True)

    results = []
    for name, picks_path in RUNS:
        mpath = TBL_DIR / f"level5_{name}_metrics.json"
        if mpath.exists():
            print(f"\n===== level5-{name}: resume skip =====", flush=True)
            results.append(json.loads(mpath.read_text())); continue
        print(f"\n===== level5-{name} (picks={picks_path}) =====", flush=True)
        try:
            results.append(train_one(name, picks_path, device, val_loader))
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    (TBL_DIR / "level5_all_metrics.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 56)
    for m in sorted(results, key=lambda x: x["final_val_avg_mf1"], reverse=True):
        print(f"{m['name']:12s} n={m['n_extra']:4d} best={m['best_val_avg_mf1']:.4f} final={m['final_val_avg_mf1']:.4f}")


if __name__ == "__main__":
    main()
