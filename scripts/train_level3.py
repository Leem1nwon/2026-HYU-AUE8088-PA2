"""Level 3 — Imbalance & Augmentation ablation on the best backbone (ViT-S/16).

Every run starts from the SAME ImageNet-remapped ViT init (clean ablation) and
fine-tunes with one imbalance strategy. The single best run becomes
``checkpoints/level3_best.pth`` (= Level 5 base model).

Three branches, isolated then combined (12 runs):

  baseline                  plain CE · normal sampling · base aug   (= Level 2 best config)
  --- Loss-level (isolate) -------------------------------------------------------------
  wce                       inverse-freq Weighted CE  (all 3 attrs)
  focal                     Focal gamma=2             (all 3 attrs)
  cb                        Class-Balanced beta=.9999 (all 3 attrs)
  ldam                      LDAM max_m=.5 s=30        (all 3 attrs)
  --- Sampling-level -------------------------------------------------------------------
  sampler-weather           plain CE + class-balanced sampler over weather
  sampler-joint             plain CE + joint (3-attr) balanced sampler
  --- Augmentation-level ---------------------------------------------------------------
  randaug                   plain CE + RandAugment
  mixup-cutmix              plain CE + Mixup/CutMix (3-head label mixing)
  --- Combinations ---------------------------------------------------------------------
  focal+sampler             Focal + sampler(weather)
  cb+sampler+randaug        CB + sampler(weather) + RandAugment
  perattr+sampler+randaug   weather=LDAM, scene=CB, timeofday=CE + sampler + RandAug

HP (ViT best, from Level 2): AdamW lr=1e-4 wd=5e-2, CosineAnnealing, epochs=25,
batch=64, AMP fp16, seed 42. Checkpoints saved fp32 (T4-load compatible).

Skeleton-bug note: weather has foggy=0 train samples. Passing raw weather counts
to ClassBalancedLoss/LDAMLoss blows up (effective-number -> 0 -> weight 1e4;
1/sqrt(sqrt(0)) -> inf margin -> NaN). We clamp weather counts to min=1 before
building weight-based losses. (foggy still cannot be learned — see report.)

wandb: own account only (WANDB_API_KEY injected). Logging matches
notebooks/level3_imbalance.ipynb exactly (run_name=f"level3-{name}",
tags=["level3", name], per-epoch metrics via trainer, post-train confusion
matrices + per-class P/R/F1 tables).

Run:
  WANDB_API_KEY=$(cat ~/.wandb_key) CUDA_VISIBLE_DEVICES=0 \
    /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/train_level3.py
  # smoke (1 epoch, 4 representative specs):
  L3_SMOKE=1 CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python scripts/train_level3.py
"""
from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.vit_load_pretrained import load_pretrained_vit
from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.datasets.samplers import class_balanced_sampler, joint_class_balanced_sampler
from src.losses.imbalanced import (
    ClassBalancedLoss,
    FocalLoss,
    LDAMLoss,
    weighted_cross_entropy,
)
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
from src.utils.trainer import MixupTrainer, MultiTaskTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform, train_transform_randaug
from src.utils.wandb_logger import WandbLogger

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 64
NUM_WORKERS = 4       # was 8 — fewer workers to avoid the DataLoader shutdown deadlock
LOADER_TIMEOUT = 180  # seconds; turns a worker hang into a fast error instead of a silent stall
EPOCHS = 25
LR = 1e-4          # ViT-pretrained best LR (Level 2)
WD = 5e-2
CKPT_DIR = Path("checkpoints")
TBL_DIR = Path("tables")
CKPT_DIR.mkdir(exist_ok=True)
TBL_DIR.mkdir(exist_ok=True)

SMOKE = os.environ.get("L3_SMOKE", "").strip().lower() in ("1", "true", "yes", "on")
if SMOKE:
    EPOCHS = 1

WANDB_PROJECT = None if SMOKE else "aue8088-pa2"
WANDB_TAGS = ["level3"]

# (name, loss_kind, sampler, aug)
#   loss_kind: plain | wce | focal | cb | ldam | perattr
#   sampler:   None | "weather" | "joint"
#   aug:       "basic" | "randaug" | "mix"
SPECS = [
    ("baseline",                "plain",   None,       "basic"),
    ("wce",                     "wce",     None,       "basic"),
    ("focal",                   "focal",   None,       "basic"),
    ("cb",                      "cb",      None,       "basic"),
    ("ldam",                    "ldam",    None,       "basic"),
    ("sampler-weather",         "plain",   "weather",  "basic"),
    ("sampler-joint",           "plain",   "joint",    "basic"),
    ("randaug",                 "plain",   None,       "randaug"),
    ("mixup-cutmix",            "plain",   None,       "mix"),
    ("focal+sampler",           "focal",   "weather",  "basic"),
    ("cb+sampler+randaug",      "cb",      "weather",  "randaug"),
    ("perattr+sampler+randaug", "perattr", "weather",  "randaug"),
]
SMOKE_SPECS = {"baseline", "ldam", "sampler-joint", "mixup-cutmix"}


def stem_of(name: str) -> str:
    return "level3_" + name.replace("+", "_").replace("-", "_")


def counts_clamped(ds: BDDAttrDataset, attr: str) -> torch.Tensor:
    """Per-class counts with min=1 (foggy=0 would break CB/LDAM weighting)."""
    return ds.class_counts(attr).clamp(min=1)


def build_losses(kind: str, ds: BDDAttrDataset, device: torch.device) -> dict:
    if kind == "plain":
        fns = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    elif kind == "wce":
        fns = {a: weighted_cross_entropy(counts_clamped(ds, a)) for a in ATTRIBUTES}
    elif kind == "focal":
        fns = {a: FocalLoss(gamma=2.0) for a in ATTRIBUTES}
    elif kind == "cb":
        fns = {a: ClassBalancedLoss(counts_clamped(ds, a), beta=0.9999) for a in ATTRIBUTES}
    elif kind == "ldam":
        fns = {a: LDAMLoss(counts_clamped(ds, a), max_m=0.5, s=30.0) for a in ATTRIBUTES}
    elif kind == "perattr":
        fns = {
            "weather": LDAMLoss(counts_clamped(ds, "weather"), max_m=0.5, s=30.0),
            "scene": ClassBalancedLoss(counts_clamped(ds, "scene"), beta=0.9999),
            "timeofday": nn.CrossEntropyLoss(),
        }
    else:
        raise ValueError(kind)
    return {a: fns[a].to(device) for a in ATTRIBUTES}


def loss_desc(kind: str) -> dict | str:
    return {
        "plain": "ce",
        "wce": "weighted_ce",
        "focal": "focal_g2.0",
        "cb": "cb_beta0.9999",
        "ldam": "ldam_m0.5_s30",
        "perattr": {"weather": "ldam", "scene": "cb", "timeofday": "ce"},
    }[kind]


def make_train_loader(ds: BDDAttrDataset, sampler_kind, g: torch.Generator) -> DataLoader:
    common = dict(batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=True,
                  worker_init_fn=seed_worker, generator=g, timeout=LOADER_TIMEOUT,
                  persistent_workers=True)
    if sampler_kind is None:
        return DataLoader(ds, shuffle=True, **common)
    if sampler_kind == "weather":
        sampler = class_balanced_sampler(ds, attribute="weather")
    elif sampler_kind == "joint":
        sampler = joint_class_balanced_sampler(ds, mode="mean")
    else:
        raise ValueError(sampler_kind)
    return DataLoader(ds, sampler=sampler, **common)


def train_one(spec, device, val_loader):
    name, loss_kind, sampler_kind, aug = spec
    stem = stem_of(name)
    set_seed(SEED, deterministic=True)

    # dataset/transform (rebuilt per spec so RandAugment + sampler counts are correct)
    tf = train_transform_randaug() if aug == "randaug" else train_transform()
    train_ds = BDDAttrDataset(DATA_ROOT, "train", transform=tf)
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = make_train_loader(train_ds, sampler_kind, g)

    # model: fresh ViT, ImageNet-remapped init
    model = vit_small_patch16_224().to(device)
    rep = load_pretrained_vit(model, verbose=False)

    loss_fns = build_losses(loss_kind, train_ds, device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WD, amp=True)

    logger = WandbLogger(
        project=WANDB_PROJECT,
        run_name=f"level3-{name}",
        config={
            "backbone": "vit_small_patch16_224", "pretrained": True,
            "loss": loss_desc(loss_kind),
            "sampler": sampler_kind or "none",
            "augment": aug,
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WD,
            "seed": SEED, "amp": "fp16",
            "init_backbone_keys": rep["n_loaded"],
        },
        tags=WANDB_TAGS + [name],
    )

    TrainerCls = MixupTrainer if aug == "mix" else MultiTaskTrainer
    trainer = TrainerCls(model, optim, sched, loss_fns, device, cfg, logger=logger)
    history = trainer.fit(train_loader, val_loader)

    # Deploy the best-Avg-MF1 epoch (captured during fit; loop still ran to the end).
    # All val analysis below is on these deployed weights -> headline, per-attr, and
    # per-class are mutually coherent and reproduce from the saved checkpoint.
    best_state = trainer.best_state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    val_pred, _, val_tgt, _ = collect_predictions(model, val_loader, device)
    cms = confusion_matrices(val_pred, val_tgt)
    prf = per_class_prf(val_pred, val_tgt)
    dep_avg = average_macro_f1(val_pred, val_tgt)        # == trainer.best_val_avg_mf1
    dep_per = per_attribute_macro_f1(val_pred, val_tgt)

    for a in ATTRIBUTES:
        logger.log_confusion_matrix(f"best/cm_{a}", cms[a], CLASS_NAMES[a])
        rows = list(zip(prf[a]["class"], prf[a]["precision"], prf[a]["recall"],
                        prf[a]["f1"], prf[a]["support"]))
        logger.log_table(f"best/prf_{a}", ["class", "P", "R", "F1", "support"],
                         [list(r) for r in rows])
    logger.finish()

    best_hist = trainer.best_val_avg_mf1
    metrics = {
        "name": name, "stem": stem,
        "loss": loss_kind, "sampler": sampler_kind, "aug": aug,
        "best_val_avg_mf1": best_hist,          # max over epochs (= deployed checkpoint)
        "best_epoch": trainer.best_epoch,
        "final_val_avg_mf1": dep_avg,           # deployed (best-epoch) eval; matches per-class below
        "final_per_mf1": dep_per,
        "prf": prf,
        "last_epoch_avg_mf1": history["val_avg_mf1"][-1],  # literal final epoch (reference only)
    }
    # SMOKE never persists artifacts — otherwise its 1-epoch results poison resume.
    if not SMOKE:
        torch.save(
            {"state_dict": best_state, "history": history, "seed": SEED,
             "best_epoch": trainer.best_epoch,
             "spec": {"name": name, "loss": loss_kind, "sampler": sampler_kind, "aug": aug},
             "lr": LR, "wd": WD},
            CKPT_DIR / f"{stem}.pth",
        )
        (CKPT_DIR / f"{stem}_history.json").write_text(json.dumps(history, indent=2))
        (TBL_DIR / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{name}] DONE  best(ep{trainer.best_epoch})={best_hist:.4f}  "
          f"last={history['val_avg_mf1'][-1]:.4f}  "
          f"per={ {k: round(v,3) for k,v in dep_per.items()} }\n", flush=True)

    # explicit teardown — release DataLoader workers + GPU mem before next run
    del trainer, train_loader, train_ds, model, optim, sched, loss_fns
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [s for s in SPECS if (s[0] in SMOKE_SPECS)] if SMOKE else SPECS
    print(f"device={device} | epochs={EPOCHS} batch={BATCH} lr={LR} wd={WD} | "
          f"project={WANDB_PROJECT} | SMOKE={SMOKE} | {len(specs)} runs", flush=True)

    val_loader = DataLoader(
        BDDAttrDataset(DATA_ROOT, "val", transform=eval_transform()),
        batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
        timeout=LOADER_TIMEOUT, persistent_workers=True,
    )

    results = []
    for spec in specs:
        name = spec[0]
        mpath = TBL_DIR / f"{stem_of(name)}_metrics.json"
        if mpath.exists() and not SMOKE:
            print(f"\n===== level3-{name}: metrics 존재 → resume skip =====", flush=True)
            results.append(json.loads(mpath.read_text()))
            continue
        print(f"\n===== level3-{name}  (loss={spec[1]}, sampler={spec[2]}, aug={spec[3]}) =====",
              flush=True)
        try:
            results.append(train_one(spec, device, val_loader))
        except Exception as e:  # one run failing must not kill the whole sweep
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    # rank by best-over-epochs val Avg-MF1; copy winner -> level3_best.pth
    results.sort(key=lambda m: m["best_val_avg_mf1"], reverse=True)
    (TBL_DIR / "level3_all_metrics.json").write_text(json.dumps(results, indent=2))

    print("=" * 64)
    for m in results:
        print(f"{m['name']:26s} best={m['best_val_avg_mf1']:.4f}  final={m['final_val_avg_mf1']:.4f}")

    if not SMOKE:
        winner = results[0]
        wstem = winner["stem"]
        shutil.copyfile(CKPT_DIR / f"{wstem}.pth", CKPT_DIR / "level3_best.pth")
        shutil.copyfile(TBL_DIR / f"{wstem}_metrics.json", TBL_DIR / "level3_best_metrics.json")
        print(f"\nbest = {winner['name']} ({winner['best_val_avg_mf1']:.4f}) "
              f"-> copied to level3_best.pth", flush=True)


if __name__ == "__main__":
    main()
