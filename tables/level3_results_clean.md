# Level 3 — Imbalance & Augmentation: 12-run ablation

ViT-S/16 (ImageNet-pretrained) backbone · AdamW lr 1e-4 / wd 5e-2 · 25 ep · seed 42 · Set A val.
**모든 수치는 best-Avg-MF1 epoch 기준** (per-attribute 3개의 평균 = Avg-MF1, 검산 일치).

| Technique | Loss | Sampler | Aug | Avg-MF1 | Δ base | weather | scene | timeofday |
|---|---|---|---|---:|---:|---:|---:|---:|
| **Baseline** | | | | | | | | |
| baseline (ref) | — | — | — | 0.7249 | ref | 0.624 | 0.704 | 0.846 |
| **Loss-level** | | | | | | | | |
| ldam | LDAM | — | — | 0.7200 | -0.0050 | 0.588 | 0.710 | 0.862 |
| wce | Weighted CE | — | — | 0.7190 | -0.0060 | 0.628 | 0.681 | 0.848 |
| focal | Focal | — | — | 0.7171 | -0.0078 | 0.602 | 0.718 | 0.831 |
| cb | Class-Balanced | — | — | 0.7161 | -0.0088 | 0.613 | 0.690 | 0.846 |
| **Sampling-level** | | | | | | | | |
| sampler joint | — | Joint (W×T) | — | 0.7195 | -0.0054 | 0.630 | 0.708 | 0.820 |
| sampler weather | — | Weather-bal. | — | 0.7065 | -0.0185 | 0.624 | 0.679 | 0.817 |
| **Augmentation** | | | | | | | | |
| mixup cutmix ⭐best | — | — | Mixup+CutMix | 0.7301 | +0.0051 | 0.631 | 0.692 | 0.867 |
| randaug | — | — | RandAugment | 0.7256 | +0.0006 | 0.612 | 0.708 | 0.857 |
| **Combined (2+ axes)** | | | | | | | | |
| focal sampler | Focal | Weather-bal. | — | 0.7236 | -0.0013 | 0.606 | 0.718 | 0.846 |
| perattr sampler randaug | Per-attr† | Weather-bal. | RandAugment | 0.7126 | -0.0124 | 0.585 | 0.706 | 0.847 |
| cb sampler randaug | Class-Balanced | Weather-bal. | RandAugment | 0.7100 | -0.0150 | 0.613 | 0.693 | 0.824 |

### Legend
- **Weighted CE** = 클래스 빈도 역가중 Cross-Entropy · **Focal** = (1−p)^γ down-weighting · **LDAM** = label-distribution-aware margin · **Class-Balanced** = effective-number 재가중.
- **Sampler** — *Weather-bal.* = weather 빈도 역수 샘플링 · *Joint (W×T)* = weather×timeofday 결합 균형.
- **Aug** — *RandAugment* = 자동 증강 정책 · *Mixup+CutMix* = 3-head 라벨 혼합 확장.
- **†Per-attr** = 속성별 다른 loss (weather:LDAM / scene:Class-Balanced / timeofday:CE).
- **Δ base** = Avg-MF1 − baseline(0.7249 plain CE).

⭐ **best = mixup cutmix** (Avg-MF1 0.7301) → level3_best.pth, Level 5 base 모델.