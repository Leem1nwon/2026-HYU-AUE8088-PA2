"""Build a Kaggle submission CSV from a checkpoint's predictions on Set A test.

Default model = checkpoints/level3_best.pth (current best, mixup-cutmix ViT).
Inference path is fp32 (T4-compatible); argmax results are hardware-independent.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/ailab/anaconda3/envs/aue8088-pa2/bin/python \
    scripts/make_submission.py [ckpt.pth] [out.csv] [arch]
  # arch: vit (default) | resnet18 | resnet50 | vgg16
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import BDDAttrDataset
from src.models.resnet import resnet18, resnet50
from src.models.vgg import VGG16
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import collect_predictions
from src.utils.submission import write_submission
from src.utils.transforms import eval_transform

ARCHES = {"vit": vit_small_patch16_224, "resnet18": resnet18,
          "resnet50": resnet50, "vgg16": VGG16}


def main() -> None:
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/level3_best.pth")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("submission/level3_best.csv")
    arch = sys.argv[3] if len(sys.argv) > 3 else "vit"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHES[arch]().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.float().to(device).eval()
    spec = ckpt.get("spec", {})
    print(f"loaded {ckpt_path} (arch={arch}, spec={spec})")

    test_loader = DataLoader(
        BDDAttrDataset("data/set_a", "test", transform=eval_transform()),
        batch_size=64, shuffle=False, num_workers=8, pin_memory=True,
    )
    preds, _, _, ids = collect_predictions(model, test_loader, device)
    print(f"predicted {len(ids)} test images")

    write_submission(out_path, ids, preds)
    print(f"wrote {out_path}")

    # quick sanity: class distribution of predictions
    from src.datasets.bdd_attr import (SCENE_CLASSES, TIMEOFDAY_CLASSES, WEATHER_CLASSES)
    import numpy as np
    for a, names in [("weather", WEATHER_CLASSES), ("scene", SCENE_CLASSES),
                     ("timeofday", TIMEOFDAY_CLASSES)]:
        u, c = np.unique(preds[a], return_counts=True)
        dist = {names[int(k)]: int(v) for k, v in zip(u, c)}
        print(f"  {a}: {dist}")


if __name__ == "__main__":
    main()
