"""End-to-end smoke test — CLAUDE.md Part2 §E 0순위 작업.

목적은 모델 성능이 아니라 **파이프라인 전 구간이 끝까지 에러 없이 도는가**의 검증이다.
데이터 로드 → 학습 2ep → 체크포인트 저장 → CPU 재로드 → eval → 제출 CSV 까지 한 번 관통한다.
특히 (1) 체크포인트 왕복(저장→재로드 후 예측 불변)과 (2) 제출 CSV 포맷을 확인한다.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_e2e.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, BDDAttrDataset
from src.models.resnet import resnet18
from src.utils.metrics import collect_predictions
from src.utils.seed import seed_worker, set_seed
from src.utils.submission import write_submission
from src.utils.trainer import MultiTaskTrainer, TrainConfig
from src.utils.transforms import eval_transform, train_transform

SEED = 42
DATA_ROOT = "data/set_a"
BATCH = 64
EPOCHS = 2
CKPT = Path("checkpoints/smoke_resnet18.pth")
SUB = Path("submission/smoke_submission.csv")


def make_loader(split: str, transform, shuffle: bool) -> DataLoader:
    ds = BDDAttrDataset(DATA_ROOT, split, transform=transform)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(
        ds, batch_size=BATCH, shuffle=shuffle, num_workers=4,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )
    return loader


def main() -> None:
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 1) 데이터
    train_loader = make_loader("train", train_transform(), shuffle=True)
    val_loader = make_loader("val", eval_transform(), shuffle=False)
    test_loader = make_loader("test", eval_transform(), shuffle=False)

    # 2) 모델 + 옵티마이저 + 손실 (3-task CE 합)
    model = resnet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    loss_fns = {a: nn.CrossEntropyLoss() for a in ATTRIBUTES}
    cfg = TrainConfig(epochs=EPOCHS, lr=3e-4, amp=True)
    trainer = MultiTaskTrainer(model, optimizer, None, loss_fns, device, cfg)

    # 3) 학습 (2 epoch — 끝까지 도는지만 본다)
    print("\n[학습]")
    trainer.fit(train_loader, val_loader)

    # 4) 체크포인트 저장 (CLAUDE.md §C: state_dict 만, fp32)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.float().state_dict(), "seed": SEED}, CKPT)
    print(f"\n[저장] {CKPT}  ({CKPT.stat().st_size / 1e6:.1f} MB)")

    # 5) 저장 직후 모델의 val 예측 (기준)
    preds_before, _, _, _ = collect_predictions(model, val_loader, device)

    # 6) 새 모델에 CPU 로드 → device 이동 → eval (체크포인트 왕복)
    model2 = resnet18()
    ckpt = torch.load(CKPT, map_location="cpu")
    model2.load_state_dict(ckpt["state_dict"])
    model2.to(device).eval()
    preds_after, _, _, _ = collect_predictions(model2, val_loader, device)

    # 7) 왕복 무결성: 로드 전후 argmax 예측이 완전히 같아야 한다
    same = all(np.array_equal(preds_before[a], preds_after[a]) for a in ATTRIBUTES)
    print(f"[체크포인트 왕복] 로드 전후 val 예측 동일: {same}")
    assert same, "체크포인트 저장/로드가 예측을 바꿈 — 규약 위반"

    # 8) test 예측 → 제출 CSV (test 라벨은 -1, argmax 예측만 사용)
    preds_test, _, _, ids = collect_predictions(model2, test_loader, device)
    write_submission(SUB, ids, preds_test)
    print(f"[제출] {SUB}  (행 {len(ids)})")

    # 9) CSV 포맷 검증
    head = SUB.read_text().splitlines()[:3]
    print("CSV 미리보기:")
    for line in head:
        print("   ", line)
    assert head[0] == "image_id,weather,scene,timeofday", "제출 CSV 헤더 불일치"
    assert len(SUB.read_text().splitlines()) == len(ids) + 1, "제출 CSV 행 수 불일치"

    print("\n✅ 엔드투엔드 뼈대 1회전 통과: "
          "데이터 → 학습 → 체크포인트 왕복 → eval → 제출 CSV 무결")


if __name__ == "__main__":
    main()
