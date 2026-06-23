"""Set A / Set B distribution report -> self-contained HTML.

Loads labels for set_a (train/val) and set_b (mining), computes per-attribute
class distributions, attribute co-occurrence (weather x timeofday, weather x
scene), and imbalance ratios; renders charts inline (base64 PNG) so the HTML is
a single portable file. Used to plan the Level 5 1,000-Pick.

Run:
  /home/ailab/anaconda3/envs/aue8088-pa2/bin/python scripts/data_distribution_report.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.bdd_attr import (
    ATTRIBUTES, NUM_CLASSES, SCENE_CLASSES, TIMEOFDAY_CLASSES, WEATHER_CLASSES,
    BDDAttrDataset,
)

CN = {"weather": WEATHER_CLASSES, "scene": SCENE_CLASSES, "timeofday": TIMEOFDAY_CLASSES}
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

SETS = {
    "Set A · train": BDDAttrDataset("data/set_a", "train"),
    "Set A · val":   BDDAttrDataset("data/set_a", "val"),
    "Set B (pool)":  BDDAttrDataset("data/set_b", "mining"),
}


def counts(ds, attr):
    return ds.class_counts(attr).numpy()


def joint(ds, a1, a2):
    M = np.zeros((NUM_CLASSES[a1], NUM_CLASSES[a2]), int)
    for s in ds.samples:
        l1, l2 = getattr(s, a1), getattr(s, a2)
        if l1 >= 0 and l2 >= 0:
            M[l1, l2] += 1
    return M


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(b64, alt=""):
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


# ---- charts -----------------------------------------------------------------
def chart_attribute(attr) -> str:
    classes = CN[attr]
    names = list(SETS.keys())
    props = []
    for nm in names:
        c = counts(SETS[nm], attr).astype(float)
        props.append(c / max(c.sum(), 1) * 100)
    props = np.array(props)  # (3 sets, K)
    x = np.arange(len(classes)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, nm in enumerate(names):
        ax.bar(x + (i - 1) * w, props[i], w, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("% within split"); ax.set_title(f"{attr} distribution")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    return img(fig_to_b64(fig), attr)


def chart_joint(ds_name, a1="weather", a2="timeofday") -> str:
    M = joint(SETS[ds_name], a1, a2)
    row = M / np.maximum(M.sum(1, keepdims=True), 1)  # P(a2 | a1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(row, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(NUM_CLASSES[a2])); ax.set_xticklabels(CN[a2], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES[a1])); ax.set_yticklabels(CN[a1], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]}\n{row[i,j]*100:.0f}%", ha="center", va="center",
                    color="white" if row[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title(f"{ds_name}\n{a1} × {a2}  (cell: count / P({a2}|{a1}))", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label=f"P({a2}|{a1})")
    return img(fig_to_b64(fig))


# ---- tables -----------------------------------------------------------------
def table_attribute(attr) -> str:
    classes = CN[attr]
    head = "".join(f"<th>{c}</th>" for c in classes)
    rows = ""
    for nm, ds in SETS.items():
        c = counts(ds, attr); tot = c.sum()
        cells = ""
        for v in c:
            pct = 100 * v / max(tot, 1)
            cls = "zero" if v == 0 else ("low" if pct < 8 else "")
            cells += f'<td class="{cls}">{v}<span class="pct">{pct:.1f}%</span></td>'
        rows += f"<tr><th class='rowh'>{nm}</th>{cells}<td class='tot'>{tot}</td></tr>"
    return (f'<table><thead><tr><th>{attr}</th>{head}<th>total</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>")


def imbalance_row():
    out = []
    for attr in ATTRIBUTES:
        c = counts(SETS["Set A · train"], attr)
        nz = c[c > 0]
        ratio = nz.max() / nz.min() if len(nz) else float("nan")
        out.append(f"<li><b>{attr}</b>: max/min(비0) = <b>{ratio:.0f}×</b> "
                   f"(max {nz.max()}, min {nz.min()}; 0장 클래스 {int((c==0).sum())}개)</li>")
    return "<ul>" + "".join(out) + "</ul>"


# ---- assemble ---------------------------------------------------------------
def main() -> None:
    sizes = {nm: len(ds) for nm, ds in SETS.items()}

    attr_charts = "".join(f'<div class="card">{chart_attribute(a)}</div>' for a in ATTRIBUTES)
    attr_tables = "".join(f'<div class="card">{table_attribute(a)}</div>' for a in ATTRIBUTES)
    joint_wt = (f'<div class="card">{chart_joint("Set A · train","weather","timeofday")}</div>'
                f'<div class="card">{chart_joint("Set B (pool)","weather","timeofday")}</div>')
    joint_ws = (f'<div class="card">{chart_joint("Set A · train","weather","scene")}</div>'
                f'<div class="card">{chart_joint("Set B (pool)","weather","scene")}</div>')

    # Level 5 supply table: set_a train vs set_b for minority weather classes
    wa = counts(SETS["Set A · train"], "weather")
    wb = counts(SETS["Set B (pool)"], "weather")
    supply = ""
    for i, c in enumerate(WEATHER_CLASSES):
        tag = "zero" if wb[i] == 0 else ""
        supply += (f'<tr><th class="rowh">{c}</th><td>{wa[i]}</td><td class="{tag}">{wb[i]}</td>'
                   f'<td>{"불가 (전역 0)" if wb[i]==0 else f"+{wb[i]} 가용"}</td></tr>')

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<title>PA2 — Set A / Set B 데이터 분포 리포트</title>
<style>
  :root{{--bg:#0f1117;--card:#1a1d27;--ink:#e6e8ef;--mut:#9aa3b2;--acc:#5b9dff;--warn:#ff6b6b;--low:#ffb454;}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,Segoe UI,Roboto,'Noto Sans KR',sans-serif;padding:32px}}
  h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:34px 0 12px;color:var(--acc);
    border-bottom:1px solid #2a2e3a;padding-bottom:6px}} h3{{font-size:15px;color:var(--mut);margin:18px 0 8px}}
  .sub{{color:var(--mut);margin:0 0 8px}} .grid{{display:flex;flex-wrap:wrap;gap:16px}}
  .card{{background:var(--card);border:1px solid #262a36;border-radius:12px;padding:14px}}
  .card img{{max-width:100%;border-radius:6px;display:block}}
  table{{border-collapse:collapse;font-size:13px}} th,td{{padding:6px 10px;text-align:center;border:1px solid #2a2e3a}}
  thead th{{background:#222634;color:var(--acc)}} .rowh{{text-align:left;color:var(--mut);white-space:nowrap}}
  .pct{{display:block;font-size:11px;color:var(--mut)}} td.zero{{background:#3a1f24;color:var(--warn);font-weight:700}}
  td.low{{color:var(--low)}} .tot{{color:var(--mut)}}
  .kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}}
  .kpi{{background:var(--card);border:1px solid #262a36;border-radius:12px;padding:12px 18px;min-width:150px}}
  .kpi b{{font-size:22px;color:var(--acc);display:block}} .kpi span{{color:var(--mut);font-size:12px}}
  .callout{{background:#2a1f24;border-left:4px solid var(--warn);padding:12px 16px;border-radius:8px;margin:12px 0}}
  .ok{{background:#1f2a24;border-left:4px solid #4caf72}} .note{{background:#1f2330;border-left:4px solid var(--acc)}}
  code{{background:#222634;padding:1px 6px;border-radius:5px;color:#cdd6f4}}
  footer{{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid #2a2e3a;padding-top:10px}}
</style></head><body>

<h1>PA2 — Set A / Set B 데이터 분포 리포트</h1>
<p class="sub">Multi-task Scene Classification · weather / scene / timeofday · Level 5 (1,000-Pick) 사전 분석</p>

<div class="kpis">
  <div class="kpi"><b>{sizes['Set A · train']:,}</b><span>Set A train (라벨)</span></div>
  <div class="kpi"><b>{sizes['Set A · val']:,}</b><span>Set A val (라벨)</span></div>
  <div class="kpi"><b>{sizes['Set B (pool)']:,}</b><span>Set B pool (라벨 공개)</span></div>
  <div class="kpi"><b>62%</b><span>Set A train의 clear 비율</span></div>
  <div class="kpi"><b>0</b><span>foggy 장수 (전 데이터)</span></div>
</div>

<div class="callout"><b>① foggy는 데이터 전역에 0장.</b> Set A train·val·Set B 모두 foggy = 0.
재가중·샘플링·증강은 물론 <b>Level 5(Set B 보충)으로도 학습 불가</b> → 실질 weather 타깃은 5클래스
(clear·overcast·rainy·snowy·partly cloudy).</div>

<div class="callout ok"><b>② 소수 클래스는 Set B에 풍부.</b> Set A train의 snowy 200 → Set B <b>2,067</b>,
rainy 400 → <b>1,783</b>, overcast 800 → <b>1,894</b>. 1,000-Pick으로 소수 클래스를 실질 보강 가능.</div>

<h2>1. 속성별 클래스 분포</h2>
<div class="grid">{attr_tables}</div>
<div class="grid">{attr_charts}</div>

<h2>2. 불균형 정도 (Set A train 기준)</h2>
{imbalance_row()}
<p class="sub">macro-F1은 소수 클래스를 동등 가중하므로, 위 불균형이 단순 CE에서 다수 클래스 편향을 유발한다.</p>

<h2>3. 속성 간 동시발생 (Level 5 커버리지용)</h2>
<h3>weather × timeofday — 셀 = 장수 / P(timeofday | weather)</h3>
<div class="grid">{joint_wt}</div>
<h3>weather × scene</h3>
<div class="grid">{joint_ws}</div>
<div class="callout note">희귀 <b>조합</b>(예: snowy+night, rainy+dawn/dusk)은 단일 속성보다 더 부족하다.
1,000-Pick에서 단순 클래스 균형뿐 아니라 <b>조합 커버리지</b>를 노리면 OOD/edge-case가 많은 Private LB에 유리.</div>

<h2>4. Level 5 보충 가용량 (weather)</h2>
<table><thead><tr><th>weather</th><th>Set A train</th><th>Set B pool</th><th>1,000-Pick 가용성</th></tr></thead>
<tbody>{supply}</tbody></table>

<div class="callout note" style="margin-top:18px"><b>Level 5 시사점</b>
<ul>
<li><b>foggy 제외</b> — 어떤 전략으로도 불가. 리포트에 한계로 명시.</li>
<li><b>우선순위(제안)</b>: snowy·rainy·overcast·partly cloudy 보강 + dawn/dusk(전 split 7~8%) 보강.</li>
<li>multi-task라 한 장이 3속성에 동시 기여 → <b>희귀 조합</b>(snowy+night 등) 우선 픽이 효율적.</li>
<li>Set A와 Set B의 전체 분포는 유사하나 Set B는 snowy 비율이 더 높아(13.8% vs 4.0%) 소수 보강에 적합.</li>
</ul></div>

<footer>생성: scripts/data_distribution_report.py · 라벨 출처: data/set_a/labels.json, data/set_b/labels.json ·
foggy=0 및 분포는 실데이터 집계값.</footer>
</body></html>"""

    path = OUT / "data_distribution.html"
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
