"""Level 5 step 1+2 — score Set B and select the 1,000 picks (notebook-faithful).

Follows notebooks/level5_data_mining.ipynb exactly, with the backbone set to our
actual best model (ViT, level3_best.pth) instead of the notebook's resnet18
placeholder:

  step 1  uncertainty = 1 - mean(max-softmax)   over the 3 heads  (notebook def)
  step 2  selection score = lam * uncertainty + (1 - lam) * rarity
          - rarity = mean over 3 attrs of inverse-frequency (Set A train), per-class
            normalized to [0,1]; combines the notebook's "uncertainty + is_rare_class"
            hint into a multi-task signal (hard-example mining + class/combination
            balance). foggy excluded (0 in Set B).

Outputs (notebook picks.json schema {strategy, num_picks, picks:[{...reason}]}):
  level5_picks.json                 (submission, top-1000)
  tables/level5_picks_250.json      (ablation subset)
  tables/level5_picks_500.json
  tables/level5_picks_random.json   (random-1000 baseline for DI)

Run:
  CUDA_VISIBLE_DEVICES=0 /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/level5_pick.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.bdd_attr import ATTRIBUTES, NUM_CLASSES, WEATHER_CLASSES, BDDAttrDataset
from src.models.vit import vit_small_patch16_224
from src.utils.metrics import collect_predictions
from src.utils.seed import set_seed
from src.utils.transforms import eval_transform

SEED = 42
LAM = 0.5            # uncertainty vs rarity balance
K = 1000
CKPT = "checkpoints/level3_best.pth"
TBL = Path("tables"); TBL.mkdir(exist_ok=True)
FOGGY = WEATHER_CLASSES.index("foggy")

STRATEGY = (
    "Hard-example + rarity + timeofday 균형. score = 0.5·uncertainty + 0.5·rarity. "
    "uncertainty = 1 − mean(max-softmax) (base=ViT best가 헷갈리는 샘플), "
    "rarity = Set A train 클래스 빈도 역수(속성별 [0,1] 정규화 후 3속성 평균; 소수 클래스/희귀 조합 보강). "
    "단 score top-K를 그대로 뽑으면 소수 클래스(dawn/dusk·residential 등 rarity·uncertainty 이중 최상위)가 "
    "독식하므로, weather·scene·timeofday 3속성 모두에 클래스 캡을 두고 score 순 greedy 선택하여 "
    "어느 한 클래스도 과반을 넘지 못하게 한다(multi-task 균형 설계). foggy는 Set B에도 0장이라 제외."
)


def to_pick(s, reason):
    return {"image_id": s.image_id, "weather": int(s.weather), "scene": int(s.scene),
            "timeofday": int(s.timeofday), "reason": reason}


def dump(path, picks, strategy):
    Path(path).write_text(json.dumps(
        {"strategy": strategy, "num_picks": len(picks), "picks": picks}, indent=2, ensure_ascii=False))


def main():
    set_seed(SEED, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # step 1 — score Set B with the best base model (ViT)
    model = vit_small_patch16_224().to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"])
    model.float().to(device).eval()
    set_b = BDDAttrDataset("data/set_b", "mining", transform=eval_transform())
    loader = DataLoader(set_b, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    preds_b, probs_b, _, ids_b = collect_predictions(model, loader, device)
    max_probs = np.stack([probs_b[a].max(axis=-1) for a in ATTRIBUTES], axis=1)
    uncertainty = 1.0 - max_probs.mean(axis=1)          # notebook definition
    print(f"scored {len(ids_b)} | uncertainty mean={uncertainty.mean():.3f}")

    # step 2 — rarity from Set A train inverse-frequency, normalized per attribute
    setA = BDDAttrDataset("data/set_a", "train")
    rar = {}
    for a in ATTRIBUTES:
        c = setA.class_counts(a).float()
        inv = 1.0 / c.clamp(min=1)
        rar[a] = (inv / inv.max()).numpy()              # rarest class -> 1.0
    rarity = np.array([
        np.mean([rar[a][getattr(s, a)] for a in ATTRIBUTES if getattr(s, a) >= 0])
        for s in set_b.samples
    ])

    score = LAM * uncertainty + (1 - LAM) * rarity
    foggy_mask = np.array([s.weather == FOGGY for s in set_b.samples])
    score[foggy_mask] = -1.0  # exclude foggy (none anyway)

    # timeofday-stratified: 3클래스 균등 쿼터 안에서 score 상위 → dawn/dusk 독식 방지
    tod = np.array([s.timeofday for s in set_b.samples])
    n_tod = NUM_CLASSES["timeofday"]
    reason = f"score={{:.3f}} (unc·{LAM}+rar·{1-LAM}, timeofday-balanced)"

    sw = np.array([s.weather for s in set_b.samples])
    ss = np.array([s.scene for s in set_b.samples])
    order = np.argsort(-score)

    def balanced(n):
        # weather·scene·timeofday 3속성 모두 클래스 캡 → score 순 greedy (한 클래스 독식 방지)
        cap_w = int(n / (NUM_CLASSES["weather"] - 1) * 1.8)   # foggy 제외 5클래스
        cap_s = int(n / NUM_CLASSES["scene"] * 1.25)
        cap_t = int(n / NUM_CLASSES["timeofday"] * 1.25)
        cw, cs, ct = {}, {}, {}
        chosen = []
        for i in order:
            if foggy_mask[i]:
                continue
            w, s, t = int(sw[i]), int(ss[i]), int(tod[i])
            if cw.get(w, 0) >= cap_w or cs.get(s, 0) >= cap_s or ct.get(t, 0) >= cap_t:
                continue
            chosen.append(i); cw[w] = cw.get(w, 0) + 1; cs[s] = cs.get(s, 0) + 1; ct[t] = ct.get(t, 0) + 1
            if len(chosen) >= n:
                break
        if len(chosen) < n:  # 캡으로 미달 시 score 순 충원
            sel = set(chosen)
            for i in order:
                if i not in sel and not foggy_mask[i]:
                    chosen.append(i)
                    if len(chosen) >= n:
                        break
        return chosen

    def picks_of(idxs):
        return [to_pick(set_b.samples[i], reason.format(score[i])) for i in idxs]

    picks_main = picks_of(balanced(K))
    dump("level5_picks.json", picks_main, STRATEGY)
    for k in (250, 500):
        dump(TBL / f"level5_picks_{k}.json", picks_of(balanced(k)), STRATEGY)

    # random-1000 baseline (DI denominator), same seed, foggy excluded
    rng = np.random.default_rng(SEED)
    pool = np.array([i for i in range(len(set_b)) if set_b.samples[i].weather != FOGGY])
    ridx = rng.choice(pool, size=K, replace=False)
    picks_rand = [to_pick(set_b.samples[i], "random baseline") for i in ridx]
    dump(TBL / "level5_picks_random.json", picks_rand, "random-1000 baseline (DI denominator)")

    # quick distribution check
    from collections import Counter
    def dist(picks, a):
        from src.utils.metrics import CLASS_NAMES
        c = Counter(p[a] for p in picks)
        return {CLASS_NAMES[a][k]: c.get(k, 0) for k in range(len(CLASS_NAMES[a]))}
    print("picks weather:", dist(picks_main, "weather"))
    print("picks scene:", dist(picks_main, "scene"))
    print("picks timeofday:", dist(picks_main, "timeofday"))
    print(f"wrote level5_picks.json (top-{K}) + 250/500 + random")


if __name__ == "__main__":
    main()
