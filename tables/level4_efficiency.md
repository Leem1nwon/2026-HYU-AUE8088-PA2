# Level 4 — Efficiency (backbone comparison)

Params/FLOPs hardware-independent. **FPS = T4 (Colab) measured.**
Avg-MF1 = best val (Level 1 for CNNs, best Level 3 for ViT).

| backbone | Params (M) | FLOPs (G) | FPS (T4) | Avg-MF1 | Pareto |
|---|---|---|---|---|---|
| vgg16 | 134.32 | 30.93 | 105.6 | 0.5629 |  |
| resnet18 | 11.18 | 3.63 | 392.3 | 0.6620 | ★ |
| resnet50 | 23.53 | 8.17 | 154.2 | 0.6244 |  |
| vit_s16 | 21.67 | 8.48 | 141.4 | 0.7301 | ★ |
