# PA2 제출물 — 2026165079 이민원

자율주행 Scene Classification (Multi-task: weather / scene / timeofday) — 단일 ViT-S/16 백본 + 3-head.
**개발/학습은 H100, 재현 검증은 Colab T4(체크포인트 eval)** 로 수행했다.

## 폴더 구조
```
notebooks/            Level 1~5 노트북 (Colab T4에서 Run All 가능)
  level1_classic_cnns.ipynb     VGG16 / ResNet-18 / ResNet-50 직접 구현
  level2_transformers.ipynb     ViT-S/16 직접 구현 (+ ImageNet pretrained remap)
  level3_imbalance.ipynb        Weighted CE/Focal/LDAM/CB · Sampler · Mixup/CutMix
  level4_xai_efficiency.ipynb   Grad-CAM · Confusion Matrix · FPS(T4)/FLOPs Pareto
  level5_data_mining.ipynb      Set B 1,000-Pick 큐레이션 + 재학습 + DI/ablation
src/                  본인 구현 소스 (모델·데이터셋·loss·augment·xai·utils)
scripts/              vit_load_pretrained.py (Level 2 ImageNet weight remap 헬퍼)
requirements.txt      Colab T4 런타임 기준 핀
level5_picks.json     ★ Level 5 제출물 (선택 1,000장 image_id + 메타데이터)
```

## 실행 방법 (Colab)
각 노트북을 **위에서부터 Run All** 하면 된다. 노트북 첫 셀들이 자동으로:
1. **GitHub repo clone** — `https://github.com/Leem1nwon/2026-HYU-AUE8088-PA2.git`
   → 실제 실행 코드는 이 clone 기준(이 폴더의 `src/`는 동일 코드의 **백업/참조본**).
2. **데이터 자동 다운로드** (gdown, 과제 공식 Drive ID `1L7YC70QlO87aIbE5lbtQ94HUINJijBKK`)
   → `../data/set_a`, `../data/set_b` 생성.
3. **체크포인트 자동 다운로드** (gdown, Drive ID `1B6XaYqchgb-wex0AHKyDJb3IF7eNcH54`)
   → `../checkpoints/*.pth` (학습 없이 **eval/그림만 재현**).

### 실행 전 확인
- **런타임 = T4 GPU** (`런타임 → 런타임 유형 변경 → T4`). 특히 Level 4 FPS 측정은 T4 기준이라야 채점 유효.
- `requirements.txt`는 Colab T4 런타임 기준으로 핀 고정. 추론 경로는 fp16/fp32(bf16 미사용).

## 재현 정책
- 학습 1시간 초과(5 Level + ablation)로 **체크포인트(.pth) 제출 → 평가 단계만 재현**(과제 허용 경로).
- 체크포인트는 `model.state_dict()`만 **fp32**로 저장, `map_location="cpu"` 로드 → GPU 종류 차이 흡수.
- 재현 기준: 동일 시드(SEED=42)에서 메트릭 ±1.0 Macro-F1 이내. 재현 대상은 재학습이 아니라 inference(argmax).

## 주요 결과 요약
- Primary metric: `Avg-MF1 = (MF1_weather + MF1_scene + MF1_timeofday) / 3`.
- **best 모델 = ViT-S/16 (Level 5 picks-1000, `level5_picks.pth`, Avg-MF1 0.7325)** — Level 3 best(mixup-cutmix, 0.7301) 초과.
- Level 4 Efficiency(T4): Pareto front = resnet18(392.3 FPS, 0.6620) · vit_s16(141.4 FPS, 0.7301).

## 비고
- `report_2026165079.pptx` 는 LMS .zip에 **별도 동봉**(이 폴더에는 미포함).
- data/ 와 checkpoints/ 는 용량 때문에 폴더에 넣지 않고 노트북에서 자동 다운로드한다.
- 모델 라이브러리(torchvision.models / timm) import 없이 백본을 직접 구현했으며, ImageNet pretrained weight만 본인 구현 state_dict에 remap했다(BDD100K pretrained 미사용).
