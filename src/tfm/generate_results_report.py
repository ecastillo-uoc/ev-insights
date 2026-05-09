"""
generate_results_report.py
──────────────────────────
Read all metadata.json files from the experiment results directory and produce
an HTML report with one table per horizon.

Rows    = dataset × model
Columns = FE-tag combinations present in the data
Cell    = MASE   (primary metric; sMAPE and RMSE shown as grey sub-text)

Colour coding (MASE)
  MASE < 0.5   → green   (beats naïve by 2×)
  MASE < 1.0   → white   (better than naïve)
  MASE < 1.5   → orange  (worse than naïve)
  MASE >= 1.5  → red     (much worse than naïve)
  missing      → light grey

Usage
  python src/tfm/generate_results_report.py
  python src/tfm/generate_results_report.py --results-dir /path/to/results --out /path/to/report.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
_RESULTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "tfm" / "doc" / "vf" / "chapters" / "results"
)
_DEFAULT_OUT = Path(__file__).resolve().parents[3] / "tfm" / "doc" / "vf" / "ev_insights_results.html"

# Canonical ordering so tables are reproducible
_DATASET_ORDER = ["Dundee", "ACN_Caltech", "ACN_JPL"]
_MODEL_ORDER = [
    "lightgbm", "xgboost",
    "lstm", "transformer", "hybrid",
    "dl_baseline_lstm", "dl_baseline_transformer", "dl_baseline_hybrid",
]
# Canonical component ordering — position in this list is the sort key within
# a tier.  Any component not listed sorts after all known ones.
_FE_COMPONENTS = ["log", "revin", "diff", "cal", "rolling"]


def _fe_sort_key(fe_tag: str) -> tuple:
    """Sort key for FE tags: (number of components, canonical index tuple).

    Splits *fe_tag* on ``'_'`` to discover its components.  Tags are ordered
    first by component count (baseline → single → pairs → triples …), then
    within each tier by the canonical position of their components in
    :data:`_FE_COMPONENTS`.  Unknown components sort after all known ones.
    """
    if not fe_tag:
        return (0, ())
    parts = fe_tag.split("_")
    indices = tuple(sorted(
        _FE_COMPONENTS.index(p) if p in _FE_COMPONENTS else len(_FE_COMPONENTS)
        for p in parts
    ))
    return (len(parts), indices)


def _fe_flags(fe_tag: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return (log, diff, cal, revin, rolling) presence flags for *fe_tag*."""
    parts = set(fe_tag.split("_")) if fe_tag else set()
    return (
        "log"     in parts,
        "diff"    in parts,
        "cal"     in parts,
        "revin"   in parts,
        "rolling" in parts,
    )
_HORIZONS = [1, 7, 30, 120]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_metadata(results_dir: Path) -> list[dict]:
    records = []
    for p in sorted(results_dir.rglob("metadata.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                records.append(json.load(fh))
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}", file=sys.stderr)
    return records


# ── Eval-strategy ordering ───────────────────────────────────────────────────
_EVAL_STRATEGY_ORDER = ["direct", "recursive", "mimo"]


# ── Image path helper ─────────────────────────────────────────────────────────

def _img_paths(ds: str, mdl: str, fe: str, h: int, eval_strategy: str) -> tuple[str, str, str]:
    """Return (avp, tts, zoom) paths relative to the HTML output file.

    case_id always includes eval_strategy (matches CaseConfig.case_id).
    model_name mirrors pipeline.py: ``{mdl}_{h}d_{ds}_{eval_strategy}``.
    Tree models (direct) call plot_test_vs_predict → no trailing eval suffix.
    Neural models (recursive/mimo) call plot_test_vs_predict_multistep → adds
    ``_{eval_strategy}`` suffix to the avp filename.
    """
    case_id    = f"{ds}_{mdl}_{fe}_{eval_strategy}"
    base       = f"chapters/results/{case_id}/{ds}"
    model_name = f"{mdl}_{h}d_{ds}_{eval_strategy}"
    # direct → plot_test_vs_predict (no extra suffix)
    # recursive / mimo → plot_test_vs_predict_multistep (adds _{eval_strategy})
    avp_suffix = "" if eval_strategy == "direct" else f"_{eval_strategy}"
    return (
        f"{base}_actual_vs_predict_{model_name}_{h}days{avp_suffix}.png",
        f"{base}_train_test_split_{model_name}_{h}days.png",
        f"{base}_train_test_split_{model_name}_{h}days_zoom.png",
    )


# ── Colour helpers ────────────────────────────────────────────────────────────

def _mase_bg(mase: float | None) -> str:
    if mase is None or math.isnan(mase):
        return "#e0e0e0"   # grey — missing
    if mase < 0.5:
        return "#b7e4b7"   # green — beats naïve by 2×
    if mase < 1.0:
        return "#f0f8f0"   # light green — better than naïve
    if mase < 1.5:
        return "#ffd580"   # orange — worse than naïve
    return "#f4b3b3"       # red — much worse than naïve


def _mase_text_color(mase: float | None) -> str:
    if mase is None or math.isnan(mase):
        return "#777"
    if mase >= 1.5:
        return "#8b0000"
    if mase >= 1.0:
        return "#7a4000"
    if mase < 0.5:
        return "#1a5c1a"
    return "#333"


def _smape_bg(smape: float | None) -> str:
    if smape is None or math.isnan(smape):
        return "#e0e0e0"   # grey — missing
    if smape < 15.0:
        return "#b7e4b7"   # green — excellent
    if smape < 25.0:
        return "#f0f8f0"   # light green — good
    if smape < 40.0:
        return "#ffd580"   # orange — poor
    return "#f4b3b3"       # red — bad


def _smape_text_color(smape: float | None) -> str:
    if smape is None or math.isnan(smape):
        return "#777"
    if smape >= 40.0:
        return "#8b0000"
    if smape >= 25.0:
        return "#7a4000"
    if smape < 15.0:
        return "#1a5c1a"
    return "#333"


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
       background: #f5f5f5; color: #222; padding: 16px; }
h1 { font-size: 1.4em; margin-bottom: 8px; }
h2 { font-size: 1.1em; margin: 20px 0 6px; color: #444; }
p.subtitle { color: #666; font-size: 0.9em; margin-bottom: 14px; }

.legend { display: flex; gap: 16px; margin-top: 8px; margin-bottom: 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.9em; }
.legend-swatch { width: 18px; height: 18px; border: 1px solid #bbb; border-radius: 3px; }

.horizon-section { margin-bottom: 32px; overflow-x: auto; }

table { border-collapse: collapse; white-space: nowrap; font-size: 11.5px; }
th, td { border: 1px solid #ccc; padding: 4px 7px; text-align: center; }
th { background: #2c3e50; color: #fff; font-weight: 600; position: sticky; top: 0; }
th.row-header { background: #34495e; text-align: left; }
td.row-header { background: #ecf0f1; font-weight: 600; text-align: left; white-space: nowrap; }
td.dataset-label { background: #d5dbdb; font-weight: 700; text-align: left;
                   border-right: 2px solid #888; }

.metric-main { font-size: 12px; font-weight: 700; display: block; }
.metric-sub  { font-size: 10px; color: #555; display: block; margin-top: 1px; }

td.clickable { cursor: zoom-in; transition: outline 0.1s; }
td.clickable:hover { outline: 2px solid #2c3e50; outline-offset: -2px; }

tr:hover td { filter: brightness(0.94); }

.fe-col-missing { background: #f5f5f5 !important; }
.no-data { color: #aaa; font-size: 10px; }

/* ── Best-in-class highlight ── */
td.best-cell { outline: 2px solid #c0932a; outline-offset: -2px; position: relative; }
td.best-cell::after {
  content: "★";
  position: absolute; top: 1px; right: 3px;
  font-size: 9px; color: #c0932a; line-height: 1;
}

.summary-bar { display: flex; gap: 24px; margin-bottom: 0; flex-wrap: wrap; }
.summary-card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
                padding: 10px 16px; text-align: center; min-width: 100px; }
.summary-card .num { font-size: 1.6em; font-weight: 700; }
.summary-card .lbl { font-size: 0.8em; color: #666; }

.sticky-header {
  position: sticky; top: 0; z-index: 200;
  background: #f5f5f5; padding: 12px 0 10px;
  border-bottom: 2px solid #ddd; margin-bottom: 16px;
}

/* ── Lightbox ── */
#lb-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.72); z-index: 9999;
  align-items: center; justify-content: center;
}
#lb-overlay.open { display: flex; }
#lb-box {
  background: #fff; border-radius: 10px; display: flex; flex-direction: column;
  max-width: 92vw; max-height: 92vh; overflow: hidden;
  box-shadow: 0 12px 48px rgba(0,0,0,.5);
}
#lb-header {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; background: #2c3e50; color: #fff;
  font-weight: 700; font-size: 13px; flex-shrink: 0;
}
#lb-close {
  margin-left: auto; background: none; border: none;
  color: #fff; font-size: 20px; cursor: pointer; line-height: 1; padding: 0 4px;
}
#lb-tabs {
  display: flex; gap: 4px; padding: 8px 12px;
  background: #ecf0f1; border-bottom: 1px solid #ccc; flex-shrink: 0;
}
.lb-tab {
  padding: 4px 14px; border: 1px solid #bbb; border-radius: 4px;
  cursor: pointer; background: #fff; font-size: 11.5px; font-weight: 600; color: #444;
}
.lb-tab.active { background: #2c3e50; color: #fff; border-color: #2c3e50; }
#lb-img-wrap {
  flex: 1; overflow: auto; padding: 12px; text-align: center;
  display: flex; align-items: center; justify-content: center;
}
#lb-img { max-width: 100%; max-height: calc(92vh - 130px); object-fit: contain; }
#lb-missing { display: none; color: #999; font-style: italic; font-size: 12px; }
#lb-nav {
  display: flex; align-items: center; justify-content: center; gap: 14px;
  padding: 8px; background: #f5f5f5; border-top: 1px solid #ddd;
  flex-shrink: 0; font-size: 12px;
}
.lb-nav-btn {
  padding: 3px 14px; border: 1px solid #bbb; border-radius: 4px;
  cursor: pointer; background: #fff; font-size: 12px; font-weight: 600;
}
.lb-nav-btn:disabled { opacity: 0.35; cursor: default; }
#lb-nav-label { min-width: 60px; text-align: center; font-weight: 700; }
"""


def _fe_label(fe_tag: str) -> str:
    return fe_tag if fe_tag else "(baseline)"


def _cell(
    mase: float | None,
    rmse: float | None,
    smape: float | None,
    ds: str = "", mdl: str = "", fe: str = "", h: int = 0,
    eval_strategy: str = "",
    is_best: bool = False,
) -> str:
    is_dundee = (ds == "Dundee")
    if is_dundee:
        bg = _smape_bg(smape)
        tc = _smape_text_color(smape)
        primary_missing = smape is None or math.isnan(smape)
    else:
        bg = _mase_bg(mase)
        tc = _mase_text_color(mase)
        primary_missing = mase is None or math.isnan(mase)
    if primary_missing:
        return f'<td style="background:{bg}"><span class="no-data">—</span></td>'
    rmse_str  = f"{rmse:.1f}"   if rmse  is not None and not math.isnan(rmse)  else "—"
    smape_str = f"{smape:.1f}%" if smape is not None and not math.isnan(smape) else "—"
    mase_str  = f"{mase:.3f}"   if mase  is not None and not math.isnan(mase)  else "—"
    avp, tts, zoom = _img_paths(ds, mdl, fe, h, eval_strategy) if ds else ("", "", "")
    title = f"{ds} / {mdl} / {fe or '(baseline)'} / {eval_strategy} — {h}d"
    extra_class = " best-cell" if is_best else ""
    data = (
        f' class="clickable{extra_class}"'
        f' data-case="{ds}_{mdl}_{fe}_{eval_strategy}"'
        f' data-horizon="{h}"'
        f' data-avp="{avp}"'
        f' data-tts="{tts}"'
        f' data-zoom="{zoom}"'
        f' data-title="{title}"'
    )
    if is_dundee:
        mase_color = _mase_text_color(mase)
        sub_line = (
            f'RMSE {rmse_str}'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{mase_color};font-weight:600">MASE {mase_str}</span>'
        )
        return (
            f'<td{data} style="background:{bg};color:{tc}">'
            f'<span class="metric-main">{smape_str}</span>'
            f'<span class="metric-sub">{sub_line}</span>'
            f'</td>'
        )
    else:
        smape_color = _smape_text_color(smape)
        sub_line = (
            f'RMSE {rmse_str}'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{smape_color};font-weight:600">sMAPE {smape_str}</span>'
        )
        return (
            f'<td{data} style="background:{bg};color:{tc}">'
            f'<span class="metric-main">{mase_str}</span>'
            f'<span class="metric-sub">{sub_line}</span>'
            f'</td>'
        )


def build_html(records: list[dict]) -> str:
    # ── Index data: (dataset, model, fe_tag, eval_strategy, horizon) → (mase, rmse, smape)
    data: dict[tuple, tuple[float | None, float | None, float | None]] = {}
    fe_tags_seen: set[str] = set()
    datasets_seen: set[str] = set()
    models_seen: set[str] = set()
    eval_strategies_seen: set[str] = set()

    for rec in records:
        ds  = rec.get("dataset", "")
        mdl = rec.get("model", "")
        fe  = rec.get("fe_tag", "")
        ev  = rec.get("eval_strategy", "direct")  # default for backward compat
        datasets_seen.add(ds)
        models_seen.add(mdl)
        fe_tags_seen.add(fe)
        eval_strategies_seen.add(ev)
        metrics = rec.get("metrics", {})
        for h_key, m in metrics.items():
            h     = int(h_key.rstrip("d"))
            mase  = m.get("MASE")
            rmse  = m.get("RMSE")
            smape = m.get("SMAPE")
            data[(ds, mdl, fe, ev, h)] = (mase, rmse, smape)

    # Ordered columns (only those with at least one result).
    # Sorted dynamically by component count then canonical component order.
    fe_cols = sorted(fe_tags_seen, key=_fe_sort_key)

    datasets = [d for d in _DATASET_ORDER if d in datasets_seen] + \
               sorted(datasets_seen - set(_DATASET_ORDER))
    models   = [m for m in _MODEL_ORDER if m in models_seen] + \
               sorted(models_seen - set(_MODEL_ORDER))
    eval_strategies = [e for e in _EVAL_STRATEGY_ORDER if e in eval_strategies_seen] + \
                      sorted(eval_strategies_seen - set(_EVAL_STRATEGY_ORDER))

    # ── Best (model, fe) per (dataset, horizon, eval_strategy) by primary metric
    # key: (ds, h, ev) → (best_mdl, best_fe)
    best_combo: dict[tuple, tuple[str, str]] = {}
    for ds in datasets:
        for h in _HORIZONS:
            for ev in eval_strategies:
                is_dundee = (ds == "Dundee")
                best_val: float | None = None
                best_pair: tuple[str, str] | None = None
                for mdl in models:
                    for fe in fe_cols:
                        entry = data.get((ds, mdl, fe, ev, h))
                        if entry is None:
                            continue
                        mase, _rmse, smape = entry
                        primary = smape if is_dundee else mase
                        if primary is None or math.isnan(primary):
                            continue
                        if best_val is None or primary < best_val:
                            best_val = primary
                            best_pair = (mdl, fe)
                if best_pair is not None:
                    best_combo[(ds, h, ev)] = best_pair

    # ── Summary stats (MASE-based)
    total_cases  = len(records)
    valid_mases  = [v[0] for v in data.values() if v[0] is not None and not math.isnan(v[0])]
    avg_mase     = sum(valid_mases) / len(valid_mases) if valid_mases else float("nan")
    green_cases  = sum(1 for v in valid_mases if v < 0.5)
    ok_cases     = sum(1 for v in valid_mases if 0.5 <= v < 1.0)
    orange_cases = sum(1 for v in valid_mases if 1.0 <= v < 1.5)
    red_cases    = sum(1 for v in valid_mases if v >= 1.5)
    n_eval_strats = len(eval_strategies)

    # ── HTML assembly
    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>EV-Insights Forecast Results</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="sticky-header">
<h1>EV-Insights — Forecast Results Summary</h1>
<p class="subtitle" style="margin-bottom:4px">Author: Eva Castillo</p>
<p class="subtitle">
  {total_cases} experiment cases &nbsp;·&nbsp;
  {len(datasets)} datasets &nbsp;·&nbsp;
  {len(models)} models &nbsp;·&nbsp;
  {len(fe_cols)} FE variants &nbsp;·&nbsp;
  {len(_HORIZONS)} horizons &nbsp;·&nbsp;
  {n_eval_strats} eval strategies ({', '.join(eval_strategies)})
  &nbsp;·&nbsp; Primary metric: <strong>MASE</strong> (sMAPE for Dundee) &nbsp;·&nbsp; Secondary: <strong>RMSE + secondary metric</strong>
</p>
""")

    # Summary cards
    parts.append('<div class="summary-bar">')
    avg_mase_str = f"{avg_mase:.3f}" if not math.isnan(avg_mase) else "—"
    for num, lbl, col in [
        (total_cases,   "total cases",     "#2c3e50"),
        (avg_mase_str,  "avg MASE",        "#555"),
        (green_cases,   "MASE < 0.5",      "#1a5c1a"),
        (ok_cases,      "MASE 0.5–1.0",    "#2e7d32"),
        (orange_cases,  "MASE 1.0–1.5",    "#7a4000"),
        (red_cases,     "MASE ≥ 1.5",      "#8b0000"),
    ]:
        parts.append(
            f'<div class="summary-card">'
            f'<div class="num" style="color:{col}">{num}</div>'
            f'<div class="lbl">{lbl}</div>'
            f'</div>'
        )
    parts.append('</div>')
    parts.append("""<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#b7e4b7"></div>MASE &lt; 0.5 (beats naïve 2×)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f0f8f0"></div>0.5 ≤ MASE &lt; 1.0 (better than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ffd580"></div>1.0 ≤ MASE &lt; 1.5 (worse than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f4b3b3"></div>MASE ≥ 1.5 (much worse than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e0e0e0"></div>No result</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#fff;outline:2px solid #c0932a;outline-offset:-2px"></div>★ Best per dataset &amp; horizon</div>
</div>
""")
    parts.append('</div>')  # close sticky-header

    # One section per (horizon × eval_strategy)
    for h in _HORIZONS:
        for ev in eval_strategies:
            parts.append(f'<div class="horizon-section">')
            parts.append(
                f'<h2>Horizon: {h} day{"s" if h > 1 else ""} '
                f'&mdash; <span style="font-weight:400;color:#555">{ev.upper()}</span></h2>'
            )
            parts.append('<table>')

            # Header row
            parts.append('<thead><tr>')
            parts.append('<th class="row-header" style="min-width:220px">Dataset / Model</th>')
            for fe in fe_cols:
                parts.append(f'<th>{_fe_label(fe)}</th>')
            parts.append('</tr></thead>')

            parts.append('<tbody>')
            for ds in datasets:
                # Dataset separator row
                n_cols = 1 + len(fe_cols)
                parts.append(
                    f'<tr><td class="dataset-label" colspan="{n_cols}">{ds}</td></tr>'
                )
                for mdl in models:
                    # Only show model row if it has at least one result for this dataset+eval_strategy
                    has_any = any(
                        (ds, mdl, fe, ev, h) in data for fe in fe_cols
                    )
                    if not has_any:
                        continue

                    parts.append('<tr>')
                    parts.append(f'<td class="row-header">&nbsp;&nbsp;{mdl}</td>')
                    for fe in fe_cols:
                        key = (ds, mdl, fe, ev, h)
                        if key in data:
                            mase, rmse, smape = data[key]
                            is_best = best_combo.get((ds, h, ev)) == (mdl, fe)
                            parts.append(_cell(mase, rmse, smape, ds, mdl, fe, h, ev, is_best=is_best))
                        else:
                            parts.append('<td class="fe-col-missing"><span class="no-data">—</span></td>')
                    parts.append('</tr>')

            parts.append('</tbody></table></div>')

    # FE legend table
    parts.append('<h2>FE Variant Descriptions</h2>')
    parts.append('<table style="margin-top:6px">')
    parts.append('<thead><tr><th>Tag</th><th>log</th><th>diff</th><th>calendar</th><th>RevIN</th><th>rolling</th></tr></thead><tbody>')
    def _yn(v: bool) -> str:
        return "✓" if v else "·"

    for fe in fe_cols:
        log, diff, cal, revin, rolling = _fe_flags(fe)
        label = _fe_label(fe)
        parts.append(
            f'<tr><td style="text-align:left;font-weight:600">{label}</td>'
            f'<td>{_yn(log)}</td><td>{_yn(diff)}</td><td>{_yn(cal)}</td>'
            f'<td>{_yn(revin)}</td><td>{_yn(rolling)}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')

    # ── Lightbox modal ────────────────────────────────────────────────────
    parts.append("""
<div id="lb-overlay">
  <div id="lb-box">
    <div id="lb-header">
      <span id="lb-title">—</span>
      <button id="lb-close" title="Close (Esc)">✕</button>
    </div>
    <div id="lb-tabs">
      <button class="lb-tab active" data-type="avp">Actual vs Predicted</button>
      <button class="lb-tab" data-type="tts">Train / Test Split</button>
      <button class="lb-tab" data-type="zoom">Split Zoom</button>
    </div>
    <div id="lb-img-wrap">
      <img id="lb-img" src="" alt=""/>
      <p id="lb-missing">Image not found for this case.</p>
    </div>
    <div id="lb-nav">
      <button class="lb-nav-btn" id="lb-prev">◀ &nbsp;prev horizon</button>
      <span id="lb-nav-label"></span>
      <button class="lb-nav-btn" id="lb-next">next horizon&nbsp; ▶</button>
    </div>
  </div>
</div>
<script>
(function () {
  'use strict';
  const HORIZONS = [1, 7, 30, 120];

  // Build data index from all clickable cells
  // imgData[caseId][horizon] = { avp, tts, zoom, title }
  const imgData = {};
  document.querySelectorAll('td[data-case]').forEach(td => {
    const cid = td.dataset.case;
    const h   = parseInt(td.dataset.horizon, 10);
    if (!imgData[cid]) imgData[cid] = {};
    imgData[cid][h] = {
      avp:   td.dataset.avp,
      tts:   td.dataset.tts,
      zoom:  td.dataset.zoom,
      title: td.dataset.title,
    };
  });

  let curCase = null, curH = null, curType = 'avp';

  const overlay  = document.getElementById('lb-overlay');
  const imgEl    = document.getElementById('lb-img');
  const titleEl  = document.getElementById('lb-title');
  const missingEl= document.getElementById('lb-missing');
  const navLabel = document.getElementById('lb-nav-label');
  const prevBtn  = document.getElementById('lb-prev');
  const nextBtn  = document.getElementById('lb-next');

  function render() {
    const info = (imgData[curCase] || {})[curH];
    if (!info) return;
    titleEl.textContent = info.title;
    navLabel.textContent = curH + 'd';

    const src = info[curType] || '';
    imgEl.style.display = 'none';
    missingEl.style.display = 'none';

    if (!src) {
      missingEl.style.display = 'block';
    } else {
      imgEl.src = src;
      imgEl.onload  = () => { imgEl.style.display = 'block'; missingEl.style.display = 'none'; };
      imgEl.onerror = () => { imgEl.style.display = 'none';  missingEl.style.display = 'block'; };
    }

    const idx = HORIZONS.indexOf(curH);
    prevBtn.disabled = idx <= 0 || !(imgData[curCase] || {})[HORIZONS[idx - 1]];
    nextBtn.disabled = idx >= HORIZONS.length - 1 || !(imgData[curCase] || {})[HORIZONS[idx + 1]];
  }

  function open(caseId, h) {
    curCase = caseId; curH = h;
    render();
    overlay.classList.add('open');
  }

  document.querySelectorAll('td[data-case]').forEach(td => {
    td.addEventListener('click', () => open(td.dataset.case, parseInt(td.dataset.horizon, 10)));
  });

  document.getElementById('lb-close').addEventListener('click', () => overlay.classList.remove('open'));
  overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });

  document.addEventListener('keydown', e => {
    if (!overlay.classList.contains('open')) return;
    if (e.key === 'Escape')      overlay.classList.remove('open');
    if (e.key === 'ArrowLeft'  && !prevBtn.disabled) prevBtn.click();
    if (e.key === 'ArrowRight' && !nextBtn.disabled) nextBtn.click();
  });

  document.querySelectorAll('.lb-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lb-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      curType = btn.dataset.type;
      render();
    });
  });

  prevBtn.addEventListener('click', () => {
    const idx = HORIZONS.indexOf(curH);
    if (idx > 0) { curH = HORIZONS[idx - 1]; render(); }
  });
  nextBtn.addEventListener('click', () => {
    const idx = HORIZONS.indexOf(curH);
    if (idx < HORIZONS.length - 1) { curH = HORIZONS[idx + 1]; render(); }
  });
})();
</script>
""")

    parts.append('</body></html>')
    return "\n".join(parts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR,
        help="Directory containing per-case sub-folders with metadata.json",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output HTML file path",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    out_path: Path = args.out

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metadata from: {results_dir}")
    records = load_all_metadata(results_dir)
    print(f"Loaded {len(records)} case records.")

    if not records:
        print("[WARN] No records found — HTML will be empty.", file=sys.stderr)

    html = build_html(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
