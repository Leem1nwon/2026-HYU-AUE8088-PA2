# CLAUDE.md — PA2 작업 컨텍스트

> 이 파일은 **Claude Code가 이 레포에서 작업할 때 읽는 컨텍스트 문서**다.
> Part 1 = 과제 명세 (자체 완결). Part 2 = 사람이 내린 컴퓨트·재현성 결정사항.
> Repo: https://github.com/IRCVLab/2026-HYU-AUE8088-PA2

## TL;DR
- Multi-task(3 head) Scene Classification. Level 1~5 순차 구현.
- **개발**: 단일 H100 서버(현재 이 환경, GPU 1장만 사용, DDP 안 씀). **재현 검증**: Colab T4 (사람이 마지막에 수동 수행).
- 모델 라이브러리(`torchvision.models`, `timm`) **import 금지** — 백본을 직접 구현. 참고 타이핑은 허용.
- 마감 **6/23(화) 23:59**. 일정 촉박 → **0순위는 엔드투엔드 뼈대 1회전**(Part 2 참고).
- 체크포인트는 **T4에서 load·eval 되도록** 저장. 재현 대상은 재학습이 아니라 inference.

---

# PART 1 — 과제 명세

## 1. 개요
자율주행 환경 인식. 단일 백본 + 3개 분류 head로 **3개 속성을 동시 예측**(Multi-task). 손실 = 3 task Cross-Entropy의 합 또는 가중합.

| 속성 | 클래스 수 | 클래스 |
|---|---|---|
| weather | 6 | clear, overcast, rainy, snowy, foggy, partly cloudy |
| scene | 3 | city street, highway, residential |
| timeofday | 3 | daytime, night, dawn/dusk |

속성 간 강한 상관 존재(예: snowy+night). 모든 이미지 224×224.

## 2. 데이터셋

| 구분 | 내용 | 활용 |
|---|---|---|
| **Set A** | train(~5천, 라벨) / val(~1천, 라벨) / test(라벨 비공개), 강한 불균형 | Level 1~4 |
| **Set B** | ~1.5만 장, **라벨 공개** | Level 5 (최대 1,000장 선별) |

- 불균형: clear 60%+ vs snowy/foggy ~1%. 단순 CE는 다수 클래스 편향.
- Set A test는 이미지만 공개 → Kaggle Leaderboard로만 성능 확인.
- 다운로드: 노트북 첫 셀에서 `gdown` 자동 다운로드. `GDRIVE_FILE_ID = "1L7YC70QlO87aIbE5lbtQ94HUINJijBKK"` → 압축 해제 시 `data/set_a`, `data/set_b` 생성.
- 디렉토리:
  ```
  data/set_a/{train,val,test}/, labels.json(train+val), {train,val,test}_ids.txt
  data/set_b/{images/, labels.json(전체 라벨), metadata.json}
  ```

## 3. Level별 구현 대상 + 배점

> **Level 1~2 공통**: 사전정의 모델 라이브러리 import 금지. 공식/논문 구현을 **참고하여 직접 타이핑**.

### Level 1 — Classic CNNs (10점)
- **VGG16 + ResNet-18 + ResNet-50** 직접 구현 (`src/models/vgg.py`, `resnet.py` 의 빈 부분 채우기).
- 백본 위 3-head multi-task.
- 분석: (a) Skip Connection이 깊은 네트워크 수렴에 미치는 영향, (b) 3 task loss 가중치의 영향.
- 산출: 체크포인트, 백본별 Avg-Macro-F1 표, VGG(skip無) vs ResNet(skip有) 손실 곡선.

### Level 2 — Vision Transformers (10점)
- **ViT-S/16** 직접 구현 (`src/models/vit.py` → `vit_small_patch16_224`). **Swin-Tiny는 선택**(`swin.py`, 안 하고 ViT 심화 가능).
- ImageNet pretrained **.pth 텐서 로드 허용** — 라이브러리 import가 아니라 **본인 구현 state_dict 키에 외부 weight를 remap**. 사용 여부·출처·매칭된 키 개수를 리포트에 기재.
- 분석: CNN 대비 (a) 데이터 효율성, (b) inductive bias 부재가 소규모·불균형 데이터에 미치는 영향.
- 참고 HP: epochs 25, AdamW lr 5e-4 wd 5e-2, CosineAnnealing.

### Level 3 — Imbalance & Augmentation (15점)
아래에서 **최소 2가지 이상 조합**:
- Loss: Weighted CE / Focal / LDAM / Class-Balanced (속성별 다른 loss 가능)
- Sampling: Class-Balanced Sampler — **어느 속성 기준으로 가중치 줄지 직접 설계**(multi-task라 한 sampler로 3속성 동시 균형 불가, 충돌 처리 필요)
- Augmentation: RandAugment / Mixup / CutMix / AugMix — **Mixup·CutMix의 라벨 혼합을 3-head multi-task로 확장**하는 것이 이 레벨의 핵심 난이도
- 보고: 각 기법이 소수/다수 클래스·각 속성에 미치는 영향을 분리 분석.

### Level 4 — XAI & Efficiency (15점)
- **Grad-CAM**: 동일 이미지에 대해 3개 head가 각각 어디를 보는지 시각화(공유 백본 + head별 backprop). ViT는 Grad-CAM 변형(Attention Rollout 등) 필요할 수 있음.
- **Confusion Matrix**: 속성별 3개, 혼동 클래스 쌍과 원인(텍스처/광원) 가설.
- **Efficiency**: T4 기준 모델별 **FPS** vs Avg-Macro-F1 Pareto front. (Params·FLOPs 선택)
- 채점 포인트 = "해석의 깊이". 그림만이 아니라 서사가 필요.

### Level 5 — The 1,000-Pick (25점, ★최대 비중)
- Set B에서 **최대 1,000장**(미만 가능) 선택 → Set A에 추가 → best 모델 재학습.
- 선택 자유: Class Balancing / Hard Example Mining(uncertainty) / Diversity(Core-Set, clustering) / Pseudo-labeling AL / **속성 조합(rainy+night 등) 커버리지** / 속성 간 correlation 활용. multi-task라 **3속성 중 우선순위 결정**이 핵심.
- 파이프라인: base 모델로 Set B 전수 scoring → 선별 → 재학습 → random-1000 baseline 대비 **DI** → **250/500/1000 ablation**.
- 산출: `level5_picks.json`(선택 image_id + 메타데이터) + Curation Report(알고리즘 의사코드/다이어그램, picks 분포 시각화, DI 비교, ablation).

## 4. 메트릭
- **Primary (Kaggle)**: `Avg-MF1 = (MF1_weather + MF1_scene + MF1_timeofday) / 3`
- Secondary: mAP
- 필수 시각화(모든 Level): 속성별 정규화 Confusion Matrix 3개, per-class P·R·F1, Top-1/Worst-class Acc
- Efficiency(L4): **T4 FPS** (batch=1, 224×224, warm-up 후 평균, 단일 forward로 3 head)
- DI(L5): `DI = (Avg-MF1[picks] − Avg-MF1[random]) / Avg-MF1[random]`

## 5. Kaggle
- Public LB 40%(일반 분포) / Private LB 60%(**OOD·Edge Case**: 역광 터널, 폭설 차선, 렌즈 오염). Public 과적합 주의.
- Submission CSV 컬럼(**코드 기준 확정**): `image_id, weather, scene, timeofday` 4컬럼. 근거 = `src/utils/submission.py:write_submission`(헤더를 직접 작성). README의 `image_id, weather`는 축약 표기이고, README가 가리키는 `submission/sample_submission.csv`는 **레포에 존재하지 않음** → sample 파일 대신 `write_submission` 출력이 단일 진실 소스. 행 예: `b1c66a42-6f7d68ca, clear, city street, daytime`.
- 제출 일 5회, 최종 2개 선택. 학생이 Kaggle에 직접 제출(LMS에 CSV 별도 제출 X).

## 6. 제출물 (LMS에 .zip 1개)
1. `pa2_<학번>_<이름>.ipynb` — Level 1~5 전부, Colab T4에서 `Run All` 가능
2. `report_<학번>.pptx` — 15~25장 (표지 → Level별 핵심 결과 → L5 전략·근거 → 결론·한계)
3. `level5_picks.json`
- (보너스 +5) Set B로 SSL(SimCLR/MoCo/MAE) 사전학습 → Set A fine-tune.

## 7. 절대 금지 / 감점
| 항목 | 결과 |
|---|---|
| 모델 라이브러리 import 사용(torchvision.models, timm) | 위반 |
| BDD100K 자체 사전학습 가중치 사용 (ImageNet은 허용) | 위반 |
| Set B 라벨 역추정(timestamp/GPS 등 메타데이터 휴리스틱) | **0점** |
| 외부 추가 라벨 데이터 학습 사용 | 위반 |
| 표절 / 수강생 간 코드 공유 (외부 참조 시 출처 명시 필수) | **0점** |
| 재현 불가 | **취득 점수 50%만 인정** |

## 8. 코드 구조 (스켈레톤)
```
src/models/    vgg.py resnet.py vit.py swin.py heads.py(MultiTaskHead)        # L1~L2 백본 직접 구현(비어 있음)
src/datasets/  bdd_attr.py(BDDAttrDataset, ATTRIBUTES, WEATHER/SCENE/TIMEOFDAY_CLASSES)
               samplers.py                                                  # L3 Class-Balanced Sampler 자리
src/losses/    imbalanced.py                                                # L3 Focal / LDAM / CB-Loss 자리
src/augment/   mix.py                                                       # L3 Mixup / CutMix (3-head 라벨혼합 확장)
src/xai/       gradcam.py                                                   # L4 Grad-CAM 자리
src/utils/     seed.py(set_seed, seed_worker) transforms.py(train/eval_transform)
               trainer.py(MultiTaskTrainer, TrainConfig) wandb_logger.py(WandbLogger)
               metrics.py(collect_predictions, confusion_matrices, CLASS_NAMES, NUM_CLASSES)
               submission.py(write_submission)   # ★ write_submission은 여기. metrics.py 아님
               efficiency.py                                                # L4 FPS / FLOPs 자리
notebooks/     level1_classic_cnns ~ level5_data_mining .ipynb
requirements.txt
```
> **단일 진실 소스 = 실제 `src/` 코드.** README의 트리는 일부 낡음(예: README는 `models/transformers.py`로 적었으나 실제는 `vit.py`+`swin.py`; README는 `submission/sample_submission.csv`를 트리에 그렸으나 실제 없음). 충돌 시 코드를 따른다.
> 위 L3/L4 스켈레톤(imbalanced·samplers·mix·gradcam·efficiency)은 **빈 채로 이미 존재** → 새로 만들지 말고 그 자리를 채운다. (§F의 "L1~L5는 모델만 교체"는 L1~L2 한정 표현)
- 환경: PyTorch ≥ 2.1, Python 3.10+. torchvision은 데이터/트랜스폼만, captum(XAI), albumentations, wandb(선택).
- **스켈레톤 코드에 버그 있을 수 있고 수정 허용됨.**
- 시드 고정: `SEED = 42`, deterministic.

---

# PART 2 — 컴퓨트 & 재현성 전략 (결정 사항)

> 코랩은 느리고 세션이 끊겨 개발에 부적합 → **개발·실험은 H100에서, 재현 검증만 T4에서.**

## A. 워크플로
1. 모든 구현·학습·실험·ablation은 **단일 H100**(이 환경)에서 풀 속도로 수행.
2. best 모델들을 **`.pth`로 동결**해 제출. 노트북은 그 `.pth`를 로드해 **eval만** 수행하도록 구성.
3. 사람(사용자)이 마지막에 코랩 T4에서 노트북 `Run All`로 재현성 확인.

## B. 재현성 = 재학습이 아니라 "체크포인트 eval"
- 재현 기준: 동일 시드·환경에서 메트릭 **±1.0 Macro-F1 이내**.
- 정책상 "학습 1시간 초과 시 체크포인트(.pth) 제출 → **평가 단계만 재현**"이 허용됨. 5개 Level + ablation 누적 학습은 1시간을 충분히 초과하므로 이 경로를 정식으로 사용.
- 핵심: 조교가 재현하는 게 **inference(argmax)** 라서 GPU·batch·정밀도가 달라도 Macro-F1이 거의 안 흔들림 → ±1.0 사실상 자동 통과. 학습 궤적을 T4와 맞출 필요 없음.

## C. 체크포인트 저장/로드 규약 (반드시 준수)
- 저장은 **`model.state_dict()`만** (optimizer/DDP wrapper 제외). 단일 GPU라 `module.` prefix는 없음.
- **fp32로 저장**: H100에서 bf16/amp로 학습했더라도 가중치는 fp32로 떨어뜨려 저장(T4 로드 시 정밀도 이슈 제거).
- 로드는 `map_location="cpu"` 후 `.to(device)` — GPU 종류 차이 흡수.
- 예:
  ```python
  # save (H100)
  torch.save({"state_dict": model.float().state_dict(), "seed": 42}, "checkpoints/levelX.pth")
  # load (T4 notebook)
  ckpt = torch.load("checkpoints/levelX.pth", map_location="cpu")
  model.load_state_dict(ckpt["state_dict"])
  model.to(device).eval()
  ```

## D. T4 호환 — "추론 경로"에만 적용
개발은 H100 풀 속도로, T4 호환은 **제출 노트북의 추론 경로**에만 신경 쓴다.
- **bf16 금지**(T4 미지원): 노트북 추론·제출 코드는 fp16 또는 fp32.
- batch size: 노트북 eval 기본값은 T4 16GB 기준(예: 64~128). H100용 큰 batch 잔재로 OOM 나지 않게.
- **FPS 측정(L4)은 반드시 코랩 T4에서**. H100 FPS는 채점 무의미.
- `requirements.txt`는 **Colab 2025.07 / T4 런타임 기준으로 핀 고정**. H100용 최신 CUDA/PyTorch와 다를 수 있음.

## E. 0순위 작업 — 엔드투엔드 뼈대 1회전 (모델 구현보다 먼저)
ResNet-18을 **2~3 epoch만** 돌려 전체 파이프라인을 한 번 관통시킨다:
`데이터 다운로드 → 3-head 출력 → 학습 1 epoch → metrics → checkpoint 저장 → (T4) load → eval → submission CSV 생성`.
끝단(특히 체크포인트 왕복, 제출 포맷)이 깨지는지 지금 확인 → 이후 L1~L5는 모델만 교체. 가장 비싼 사고("마감 당일 T4에서 .pth가 안 열림")를 예방.

## F. 권장 작업 순서
1. **뼈대 1회전**(E) — 파이프라인 검증 + 체크포인트 규약 확정
2. **Level 1**(VGG/ResNet 직접 구현) → 가장 오래 걸림, 일찍 시작
3. **Level 2**(ViT) — pretrained remap 키 매핑 주의(조용히 미적재되면 디버깅 시간 소모)
4. **Level 3**(불균형) → best 모델 확정·동결
5. **Level 5**(큐레이션, 최대 배점) — base 모델 scoring부터
6. **Level 4**(XAI·FPS) — FPS만 T4
7. 리포트(.pptx) + 재현성 최종 점검(체크포인트·시드·경로·requirements)
