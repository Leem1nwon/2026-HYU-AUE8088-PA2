# Level 4 — XAI & Efficiency 결과 리포트

> PA2 Multi-task Scene Classification 통합 리포트(.pptx)용 정리본.
> 단일 백본 + 3-head(weather / scene / timeofday) 구조에 대한 해석 가능성(XAI), 혼동 패턴(Confusion Matrix), 효율성 트레이드오프(Efficiency)를 분석한다.

## 분석 포인트

본 Level의 README 분석 포인트는 다음 3가지다.

1. **XAI (Grad-CAM)** — 동일 이미지에 대해 세 head(weather / scene / timeofday)가 각각 어디를 보는지 시각화하고, multi-task 학습이 head 간 attention을 어떻게 분산시키는지 해석한다.
2. **Confusion Matrix 분석** — 속성별 3개 CM에서 어떤 클래스 쌍이 혼동되는지, 그 원인이 텍스처인지 광원인지 가설을 세운다.
3. **Efficiency Trade-off** — FPS vs Avg-MF1 Pareto front를 그리고, Params / FLOPs를 함께 첨부해 효율-정확도 균형을 논한다.

---

## 기본 결과 (Efficiency Table)

| backbone | Params (M) | FLOPs (G) | FPS (T4, 채점값) | Avg-MF1 | Pareto |
|---|---|---|---|---|---|
| vgg16 | 134.32 | 30.93 | 105.6 | 0.5629 | dominated |
| resnet18 | 11.18 | 3.63 | **392.3** | 0.6620 | ★ |
| resnet50 | 23.53 | 8.17 | 154.2 | 0.6244 | dominated |
| vit_s16† | 21.67 | 8.48 | 141.4 | **0.7301** | ★ |

† vit_s16 Avg-MF1 0.7301 = Level 3 best(mixup-cutmix, best-epoch) 기준이며 `level3_best.pth` eval로 재현된다. 순수 baseline ViT(plain CE)는 0.7249.

> **FPS = Colab T4 측정값(채점 기준, batch=1·224×224·warm-up 후 평균).** Params/FLOPs는 하드웨어 무관 고유값이다.
> **하드웨어 의존성 — 순위 역전 확인**: 동일 모델을 H100에서 측정했을 땐 vgg16이 1042.7로 **가장 빨랐으나**, T4에서는 105.6으로 **가장 느려졌다**(resnet18이 922.7→392.3으로 1위 유지). vgg16의 30.93G FLOPs·134M params(대형 FC)는 H100의 높은 처리량에 가려졌다가 **메모리 대역폭이 제한적인 T4에서 비용이 드러난 것**으로, "FPS는 반드시 채점 하드웨어(T4)에서 측정해야 한다"는 점을 실측으로 보여준다.

---

## 1. XAI (Grad-CAM)

> **head-divergence 측정**: 각 head CAM을 [0,1] 정규화 후 3개 head 쌍의 평균 절대차(mean pairwise MAD)를 산출. 값이 클수록 head들이 서로 다른 영역을 본다. **val 20장 평균(robust)**, figure는 3 예시(주간 highway·야간·dawn/dusk). 수치는 `tables/level4_cam_diff.json`.

### CNN (ResNet-18)
ResNet-18 Grad-CAM(마지막 conv = `layer4`)에서는 **head별 CAM이 명확히 분화**되었다. head 간 CAM 차이가 **mean 0.240**(20장)로 측정되어, weather / scene / timeofday 세 head가 각각 서로 다른 영역에 주목한다. "3개 head가 각각 다른 곳을 본다"는 multi-task의 의도가 시각적으로 잘 드러난다. 이는 CNN이 국소(local) convolution 연산으로 특징을 추출하기 때문에, 각 head가 자신의 속성에 유효한 국소 영역으로 분화되기 쉽기 때문으로 해석된다.

![ResNet-18 Grad-CAM (2 예시 × 3 head)](../figures/level4_gradcam_cnn.png)

### ViT (ViT-S/16) — multi-block 누적(LayerCAM 방식)
ViT는 표준 Grad-CAM이 그대로 안 통한다. **마지막 block(`blocks[-1]`)의 출력 patch 토큰은 head(CLS만 사용) 뒤에 더 이상 attention이 없어 gradient = 0** → CAM 전멸(측정: blocks[-1] patch grad L2 = 0.0). 단일 중간 block(`blocks[-4]` 등)으로 복구는 되나 신호가 약하다. 그래서 **12개 block 각각의 (grad·act) CAM을 [0,1] 정규화 후 patch별로 누적(LayerCAM 방식)** 했다 — ViT는 전 block이 동일한 14×14 토큰 격자라 정렬이 자동으로 맞고, 정규화 덕에 에너지 큰 초기 block이 독식하지 않는다. 누적으로 head-divergence가 단일 block(0.066)→**0.099**로 향상된다.

그럼에도 **head 간 CAM 차이는 mean 0.099로 CNN(0.240)의 약 40%(2.4배 낮음)** 수준이다. ViT가 CLS 토큰으로 분류하고 전역(global) self-attention을 쓰기 때문에, 3개 head가 결국 비슷한 영역을 공유하기 때문이다 — 누적으로도 못 메우는 **구조적 차이**(layer를 바꿔도 0.04~0.11에 머묾). 즉 CNN의 국소 conv가 head별 분화를 유도하는 것과 대조된다.

이 한계를 보완하기 위해 **Attention Rollout**(head-agnostic)을 보조로 사용했다 — 특정 head가 아니라 백본 전체가 주목하는 영역을 보여준다(별도 처리).

![ViT Grad-CAM 3-head (multi-block 누적, 2 예시)](../figures/level4_gradcam_vit.png)

### 해석 요약
"multi-task 학습이 head 간 attention을 분산시킨다"는 명제는 **CNN에서는 선명하게(0.240) 관찰되고, ViT에서는 누적을 써도 약하게(0.099)** 나타난다. 이 CNN vs ViT 대비가 본 Level의 핵심 해석 포인트다 — 백본의 연산 구조(국소 conv vs 전역 attention/공유 CLS)가 head별 attention 분화 정도를 결정한다. (수치·방법 출처: `tables/level4_cam_diff.json`)

---

## 2. Confusion Matrix 분석

속성별 정규화 Confusion Matrix 3개를 통해 어떤 클래스 쌍이 혼동되는지 분석한다. 아래 수치는 **best 모델(ViT, `level3_best`) 기준 worst recall과 그 혼동 누출률**이다(Set A val). 각 CM에서 강한 혼동 셀(off-diag ≥ 0.15)은 빨간 박스, worst-recall 클래스는 주황 점선으로 표시했다.

| 속성 | worst 클래스 | recall | 혼동 대상(누출률) | 가설(원인) |
|---|---|---|---|---|
| weather | snowy | 0.58 | clear로 흡수 (0.26) | 텍스처 — 눈 덮인 노면이 흐린 clear와 시각적으로 유사 |
| scene | residential | 0.36 | city street과 혼동 (0.58) | 텍스처/구조 — 두 장면의 도로·건물 구조가 유사 |
| timeofday | dawn/dusk | 0.58 | daytime 경계 (0.35) | 광원 — 여명/황혼의 광원이 낮 경계에서 모호 |

### 가설
- **weather: snowy ↔ clear** — 눈으로 덮인 노면은 텍스처가 단조롭고 밝아 흐린 clear와 구분이 어렵다. 또한 snowy는 train 분포상 소수 클래스라 다수 클래스인 clear로 흡수되는 편향이 겹친다. → **텍스처 가설.**
- **scene: residential ↔ city street** — 주거지와 도심 거리는 도로·건물·차선 등 구조적 외형이 비슷해 백본이 구별할 단서가 적다. → **텍스처/구조 가설.**
- **timeofday: dawn/dusk ↔ daytime/night** — 여명·황혼은 광원의 밝기·색온도가 낮과 밤 사이 연속선상에 있어 경계가 모호하다. → **광원 가설.**

best 모델(ViT) 속성별 정규화 Confusion Matrix(혼동 셀 빨간 박스 강조)는 아래 figure를 참조한다.

![best ViT 정규화 Confusion Matrix 3속성 (혼동 셀 빨간 박스)](../figures/level4_confusion.png)

---

## 3. Efficiency Trade-off

FPS vs Avg-MF1 Pareto front를 통해 효율-정확도 균형을 분석한다.

![FPS vs Avg-MF1 Pareto (T4)](../figures/level4_pareto.png)

### Pareto 분석 (T4 기준)
T4 FPS 순위는 **resnet18(392.3) > resnet50(154.2) > vit_s16(141.4) > vgg16(105.6)**. Pareto front(다른 모델에 두 축 모두 밀리지 않는 점)는 **resnet18·vit_s16 두 점**이고, vgg16·resnet50은 둘 다 지배(dominated)된다.

- **효율 코너 — resnet18 (Pareto)**: 11.18M params / 3.63G FLOPs로 가장 가볍고 T4에서 가장 빠르며(392.3), Avg-MF1 0.6620으로 경량 모델 중 정확도도 양호하다. 자원 제약·실시간 환경의 1순위.
- **정확도 코너 — vit_s16 (Pareto)**: Avg-MF1 0.7301로 최고 정확도. 141.4 FPS로 느리지만 resnet50과 비슷한 규모(21.67M)에서 정확도가 크게 앞선다. 정확도 우선이면 1순위.
- **dominated — resnet50**: resnet18보다 무겁고(23.53M vs 11.18M) 느린데(154.2 < 392.3) Avg-MF1도 낮아(0.6244 < 0.6620) **resnet18에 두 축 모두 밀린다**. 본 설정에서 resnet50을 쓸 이유가 없다.
- **dominated — vgg16**: 134.32M params·30.93G FLOPs로 가장 무겁고, T4에서 가장 느리며(105.6) Avg-MF1도 최저(0.5629). 모든 면에서 열위다.

### 측정 환경 주의
**FPS는 채점 하드웨어(Colab T4)에서 측정한 값**이다(batch=1, 224×224, warm-up 후 평균, 단일 forward로 3 head 동시). 위 "순위 역전"(vgg16: H100 1위 → T4 꼴찌)이 보여주듯 FPS는 하드웨어에 강하게 의존하므로 H100 값은 채점에 쓸 수 없다. Params/FLOPs는 하드웨어 무관값이라 그대로 사용한다.

---

## 통합 리포트용 핵심 메시지

- **XAI 핵심**: CNN(ResNet-18)은 국소 conv 덕에 head별 Grad-CAM이 뚜렷하게 분화(head-divergence 0.240)되어 multi-task의 attention 분산을 선명하게 보여준다. ViT는 CLS 토큰 + 전역 self-attention 때문에 head 간 영역이 공유되어, **multi-block 누적(LayerCAM)으로 끌어올려도** 0.099(CNN의 ~40%)에 그친다. 이 CNN vs ViT 대비가 본 Level의 해석 포인트다. (수치 `tables/level4_cam_diff.json`)
- **ViT Grad-CAM 구현 노트**: `blocks[-1]` 출력 patch는 gradient = 0(CLS만 분류, 이후 attention 없음 = dead-end)이라 CAM 전멸 → **12 block (grad·act)를 정규화 후 patch별 누적(LayerCAM)**으로 신호 강화(head-div 0.066→0.099). head-agnostic Attention Rollout은 보조로 사용.
- **CM 핵심 혼동 쌍**: weather snowy↔clear(텍스처), scene residential↔city street(텍스처/구조), timeofday dawn/dusk↔daytime/night(광원). best ViT에서도 동일 쌍이 약하게 잔존한다.
- **Efficiency 핵심 (T4 측정)**: Pareto front = **resnet18(392.3 FPS, 0.6620)** + **vit_s16(141.4 FPS, 0.7301)** 두 점. vgg16·resnet50은 dominated. 효율 우선이면 resnet18, 정확도 우선이면 vit_s16.
- **하드웨어 의존성 실측**: vgg16 FPS가 H100 1위(1042.7)에서 **T4 꼴찌(105.6)로 역전** — 대형 FC의 메모리 비용이 T4에서 드러남. FPS는 반드시 채점 하드웨어(T4)에서 측정해야 함을 입증.
