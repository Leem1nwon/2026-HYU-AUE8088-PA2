# Level 3 motivation — imbalance vs per-class performance

Baseline: **ViT-pretrained, plain CE (no imbalance handling)** — Avg-MF1(final) = 0.7249. Set A train N=5,000; metrics on Set A val.

| Attribute | Class | Train % | Val support | Recall | **F1** | Tier |
|---|---|---:|---:|---:|---:|---|
| weather | clear | 62% | 300 | 0.950 | 0.898 | majority |
| weather | overcast | 16% | 50 | 0.680 | 0.723 | mid |
| weather | partly cloudy | 10% | 50 | 0.640 | 0.674 | **minority** |
| weather | rainy | 8% | 50 | 0.560 | 0.667 | **minority** |
| weather | snowy | 4% | 50 | 0.720 | 0.783 | **minority** |
| weather | foggy | 0% | 0 | 0.000 | 0.000 | zero-support |
| scene | city street | 61% | 306 | 0.879 | 0.832 | majority |
| scene | highway | 28% | 141 | 0.660 | 0.710 | mid |
| scene | residential | 11% | 53 | 0.491 | 0.571 | **minority** |
| timeofday | daytime | 50% | 242 | 0.959 | 0.959 | majority |
| timeofday | night | 43% | 232 | 0.996 | 0.985 | mid |
| timeofday | dawn/dusk | 8% | 26 | 0.538 | 0.596 | **minority** |

**Pearson r (train% vs F1, foggy excluded) = 0.77** — rarer class -> lower F1. foggy (0 train) collapses to F1 = 0.