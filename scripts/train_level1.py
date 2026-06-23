"""Level 1 본 학습 — VGG16 / ResNet18 / ResNet50.

notebooks/level1_classic_cnns.ipynb 의 ``train_one`` 과 **동일한 wandb 로깅 방식**:
  - WandbLogger 로 epoch별 자동 로깅(trainer.fit) + 학습 후 속성별 confusion matrix 업로드.
  - run_name=f"level1-{name}", project="aue8088-pa2", tags=["level1", name].

계정: 본인(minwonlee) — WANDB_API_KEY 환경변수로 주입. 서버 netrc(타인 자격)는 건드리지 않음.

실행:
  WANDB_API_KEY=$(cat ~/.wandb_key) CUDA_VISIBLE_DEVICES=0 python scripts/train_level1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.resnet import resnet18, resnet50
from src.models.vgg import VGG16
from src.utils.metrics import CLASS_NAMES, collect_predictions, confusion_matrices
from src.utils.seed import seed_worker, set_seed
from src.utils.trainer import MultiTaskTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform
from src.utils.wandb_logger import WandbLogger

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 64        # 노트북 셀7과 동일
EPOCHS = 30       # 노트북 셀8과 동일
LR = 3e-4         # 노트북 셀8과 동일
WD = 5e-4
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

WANDB_PROJECT = "aue8088-pa2"   # 노트북 셀6과 동일 (끄려면 None)
WANDB_TAGS = ["level1"]

MODELS = {"resnet18": resnet18, "resnet50": resnet50, "vgg16": VGG16}


def make_loader(split: str, transform, shuffle: bool) -> DataLoader:
    ds = BDDAttrDataset(DATA_ROOT, split, transform=transform)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        ds, batch_size=BATCH, shuffle=shuffle, num_workers=8,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )


def train_one(name, model_fn, device, train_loader, val_loader):
    set_seed(SEED, deterministic=True)
    model = model_fn().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    losses = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    cfg = TrainConfig(epochs=EPOCHS, lr=LR, weight_decay=WD)

    # wandb 로거 — 노트북 셀8과 동일한 config/run_name/tags
    logger = WandbLogger(
        project=WANDB_PROJECT,
        run_name=f"level1-{name}",
        config={
            "backbone": name, "epochs": EPOCHS, "batch": BATCH,
            "lr": LR, "weight_decay": WD, "seed": SEED,
            "loss_weights": cfg.loss_weights,
        },
        tags=WANDB_TAGS + [name],
    )
    trainer = MultiTaskTrainer(model, optim, sched, losses, device, cfg, logger=logger)
    history = trainer.fit(train_loader, val_loader)

    # 학습 후 — 속성별 정규화 confusion matrix 를 wandb 에 업로드 (노트북과 동일)
    val_pred, _, val_tgt, _ = collect_predictions(model, val_loader, device)
    cms = confusion_matrices(val_pred, val_tgt)
    for a in ATTRIBUTES:
        logger.log_confusion_matrix(f"final/cm_{a}", cms[a], CLASS_NAMES[a])
    logger.finish()

    # 체크포인트: fp32 state_dict + history (T4 로드 호환)
    torch.save(
        {"state_dict": model.float().state_dict(), "history": history, "seed": SEED},
        CKPT_DIR / f"level1_{name}.pth",
    )
    (CKPT_DIR / f"level1_{name}_history.json").write_text(json.dumps(history, indent=2))
    best = max(history["val_avg_mf1"])
    print(f"[{name}] DONE  best val Avg-MF1={best:.4f}\n", flush=True)
    return best


def main() -> None:
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  | epochs={EPOCHS} batch={BATCH} lr={LR} wd={WD} | "
          f"wandb_project={WANDB_PROJECT}", flush=True)

    train_loader = make_loader("train", train_transform(), shuffle=True)
    val_loader = make_loader("val", eval_transform(), shuffle=False)

    results = {}
    for name, fn in MODELS.items():
        results[name] = train_one(name, fn, device, train_loader, val_loader)

    print("=" * 50)
    for name, best in results.items():
        print(f"{name:10s} best val Avg-MF1={best:.4f}")


if __name__ == "__main__":
    main()
