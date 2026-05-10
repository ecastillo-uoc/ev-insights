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

# Set to True to show FE columns/cases that include seasonal differencing.
# Set to False (default) to omit all diff and diff-combination columns
# from tables AND from metric averages / best-cell selection.
_INCLUDE_DIFF: bool = False


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

# ── Outlier detection ─────────────────────────────────────────────────────────
_IQR_MULTIPLIER = 1.5  # standard Tukey fence multiplier


def _iqr_upper_fence(values: list[float], multiplier: float = _IQR_MULTIPLIER) -> float:
    """Return Q3 + multiplier × IQR for *values*. Returns inf when < 4 values."""
    if len(values) < 4:
        return float("inf")
    sv = sorted(values)
    n = len(sv)

    def _quartile(q: float) -> float:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return sv[lo] + (pos - lo) * (sv[hi] - sv[lo])

    q1 = _quartile(0.25)
    q3 = _quartile(0.75)
    return q3 + multiplier * (q3 - q1)


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

def _img_paths(ds: str, mdl: str, fe: str, h: int, eval_strategy: str) -> tuple[str, str, str, str]:
    """Return (avp, tts, zoom, win) paths relative to the HTML output file.

    case_id always includes eval_strategy (matches CaseConfig.case_id).
    model_name mirrors pipeline.py: ``{mdl}_{h}d_{ds}_{eval_strategy}``.

    direct   → avp = plot_test_vs_predict (no trailing eval suffix); win = ""
    recursive/mimo → avp = plot_test_vs_predict_multistep (_{eval_strategy} suffix);
                     win = plot_test_vs_predict of a single representative window
                           (``{model_name}_window_sample``, no extra suffix).
    """
    case_id    = f"{ds}_{mdl}_{fe}_{eval_strategy}"
    base       = f"chapters/results/{case_id}/{ds}"
    model_name = f"{mdl}_{h}d_{ds}_{eval_strategy}"
    # direct → plot_test_vs_predict (no extra suffix); no window sample
    # recursive / mimo → plot_test_vs_predict_multistep (adds _{eval_strategy})
    #                    + a window-sample plot saved by plot_test_vs_predict
    if eval_strategy == "direct":
        avp = f"{base}_actual_vs_predict_{model_name}_{h}days.png"
        win = ""
    else:
        avp = f"{base}_actual_vs_predict_{model_name}_{h}days_{eval_strategy}.png"
        win = f"{base}_actual_vs_predict_{model_name}_window_sample_{h}days.png"
    return (
        avp,
        f"{base}_train_test_split_{model_name}_{h}days.png",
        f"{base}_train_test_split_{model_name}_{h}days_zoom.png",
        win,
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

/* ── Discarded / outlier cells ── */
td.discarded-cell {
  position: relative;
  background-image: repeating-linear-gradient(
    -45deg, rgba(180,0,0,.07), rgba(180,0,0,.07) 3px,
    transparent 3px, transparent 8px) !important;
}
td.discarded-cell .metric-main { opacity: 0.55; text-decoration: line-through; }
td.discarded-cell .metric-sub  { opacity: 0.45; }
.discard-badge {
  position: absolute; top: 1px; left: 3px;
  font-size: 9px; color: #c00000; line-height: 1;
  font-weight: 700; cursor: help;
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

/* ── Best summary table ── */
.best-table { border-collapse: collapse; white-space: nowrap; font-size: 11.5px; }
.best-table th { background: #2c3e50; color: #fff; padding: 6px 10px; font-weight: 600;
                 border: 1px solid #34495e; }
.best-table td { border: 1px solid #ccc; padding: 5px 10px; text-align: center; }
.best-table td.best-model { font-weight: 700; }
.best-row { cursor: pointer; }
.best-row:hover td { background: #eaf0ff !important; filter: none; }
.best-row td.metric-mase { font-weight: 700; }

/* ── Scatter modal ── */
#sc-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.72); z-index: 9998;
  align-items: center; justify-content: center;
}
#sc-overlay.open { display: flex; }
#sc-box {
  background: #fff; border-radius: 10px; display: flex; flex-direction: column;
  width: min(680px, 95vw); max-height: 90vh; overflow: hidden;
  box-shadow: 0 12px 48px rgba(0,0,0,.5);
}
#sc-header {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; background: #2c3e50; color: #fff;
  font-weight: 700; font-size: 13px; flex-shrink: 0;
}
#sc-subtitle { font-size: 10.5px; font-weight: 400; opacity: 0.75; margin-left: 4px; }
#sc-close {
  margin-left: auto; background: none; border: none;
  color: #fff; font-size: 20px; cursor: pointer; line-height: 1; padding: 0 4px;
}
#sc-plot { flex: 1; overflow: auto; padding: 10px; background: #fff; }
#sc-legend { flex-shrink: 0; border-top: 1px solid #eee; background: #f8f8f8; min-height: 36px; }

/* ── Scatter tooltip ── */
#sc-tooltip {
  position: fixed; display: none; pointer-events: none;
  background: rgba(28,28,28,.93); color: #fff;
  padding: 7px 11px; border-radius: 6px; font-size: 11px;
  line-height: 1.6; white-space: pre-line; z-index: 99999;
  max-width: 270px; box-shadow: 0 3px 12px rgba(0,0,0,.45);
}
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
    is_discarded: bool = False,
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
    avp, tts, zoom, win = _img_paths(ds, mdl, fe, h, eval_strategy) if ds else ("", "", "", "")
    title = f"{ds} / {mdl} / {fe or '(baseline)'} / {eval_strategy} — {h}d"
    extra_class = " best-cell" if is_best else ""
    if is_discarded:
        extra_class += " discarded-cell"
    data = (
        f' class="clickable{extra_class}"'
        f' data-case="{ds}_{mdl}_{fe}_{eval_strategy}"'
        f' data-horizon="{h}"'
        f' data-avp="{avp}"'
        f' data-tts="{tts}"'
        f' data-zoom="{zoom}"'
        f' data-win="{win}"'
        f' data-title="{title}"'
    )
    discard_badge = '<span class="discard-badge" title="Outlier — excluded from averages">⊘</span>' if is_discarded else ""
    if is_dundee:
        mase_color = _mase_text_color(mase)
        sub_line = (
            f'RMSE {rmse_str}'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{mase_color};font-weight:600">MASE {mase_str}</span>'
        )
        return (
            f'<td{data} style="background:{bg};color:{tc}">'
            f'{discard_badge}'
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
            f'{discard_badge}'
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
    yrange: dict[tuple, tuple[float, float]] = {}  # (ds, h) → (y_min, y_max); same across all models

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
            ymin_v, ymax_v = m.get("y_min"), m.get("y_max")
            if ymin_v is not None and ymax_v is not None and (ds, h) not in yrange:
                yrange[(ds, h)] = (float(ymin_v), float(ymax_v))

    # Ordered columns (only those with at least one result).
    # Sorted dynamically by component count then canonical component order.
    # When _INCLUDE_DIFF is False, strip any FE tag that contains "diff".
    def _fe_has_diff(tag: str) -> bool:
        return "diff" in tag.split("_") if tag else False

    fe_cols = sorted(
        (t for t in fe_tags_seen if _INCLUDE_DIFF or not _fe_has_diff(t)),
        key=_fe_sort_key,
    )

    datasets = [d for d in _DATASET_ORDER if d in datasets_seen] + \
               sorted(datasets_seen - set(_DATASET_ORDER))
    models   = [m for m in _MODEL_ORDER if m in models_seen] + \
               sorted(models_seen - set(_MODEL_ORDER))
    eval_strategies = [e for e in _EVAL_STRATEGY_ORDER if e in eval_strategies_seen] + \
                      sorted(eval_strategies_seen - set(_EVAL_STRATEGY_ORDER))

    # ── Per-dataset IQR outlier thresholds (MASE, RMSE, sMAPE) ───────────────
    # Only consider keys whose FE tag is in the active fe_cols set.
    fe_cols_set = set(fe_cols)
    active_keys = [k for k in data if k[2] in fe_cols_set]

    # A case is "discarded" if ANY metric exceeds Q3 + 1.5·IQR for its dataset.
    ds_thresholds: dict[str, dict[str, float]] = {}
    for ds in datasets:
        mase_vals  = [v[0] for k, v in data.items() if k in active_keys and k[0] == ds and v[0] is not None and not math.isnan(v[0])]
        rmse_vals  = [v[1] for k, v in data.items() if k in active_keys and k[0] == ds and v[1] is not None and not math.isnan(v[1])]
        smape_vals = [v[2] for k, v in data.items() if k in active_keys and k[0] == ds and v[2] is not None and not math.isnan(v[2])]
        ds_thresholds[ds] = {
            "mase":  _iqr_upper_fence(mase_vals),
            "rmse":  _iqr_upper_fence(rmse_vals),
            "smape": _iqr_upper_fence(smape_vals),
        }

    def _is_discarded(key: tuple) -> bool:
        """True if any metric for this key exceeds its dataset IQR upper fence."""
        ds = key[0]
        entry = data.get(key)
        if entry is None:
            return False
        mase, rmse, smape = entry
        th = ds_thresholds.get(ds, {})
        if mase  is not None and not math.isnan(mase)  and mase  > th.get("mase",  float("inf")):
            return True
        if rmse  is not None and not math.isnan(rmse)  and rmse  > th.get("rmse",  float("inf")):
            return True
        if smape is not None and not math.isnan(smape) and smape > th.get("smape", float("inf")):
            return True
        return False

    discarded_set: set[tuple] = {k for k in active_keys if _is_discarded(k)}

    # ── Best (model, fe) per (dataset, horizon, eval_strategy) ───────────────
    # Combined-metric selection: primary metric determines rank, but a result is
    # only eligible if it is NOT discarded. Within valid results the secondary
    # metrics further tiebreak/invalidate: any secondary metric that is itself
    # an IQR outlier for the dataset disqualifies the candidate.
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
                        key = (ds, mdl, fe, ev, h)
                        if key not in data or key in discarded_set:
                            continue
                        mase, _rmse, smape = data[key]
                        primary = smape if is_dundee else mase
                        if primary is None or math.isnan(primary):
                            continue
                        if best_val is None or primary < best_val:
                            best_val = primary
                            best_pair = (mdl, fe)
                if best_pair is not None:
                    best_combo[(ds, h, ev)] = best_pair

    # ── Summary stats (valid = not discarded) ─────────────────────────────────
    total_cases  = len(records)
    n_discarded  = len(discarded_set)
    valid_keys   = [k for k in active_keys if k not in discarded_set]
    valid_mases:  list[float] = []
    valid_smapes: list[float] = []
    for k in valid_keys:
        m, _r, s = data[k]
        if m is not None and not math.isnan(m):
            valid_mases.append(m)
        if s is not None and not math.isnan(s):
            valid_smapes.append(s)
    avg_mase     = sum(valid_mases)  / len(valid_mases)  if valid_mases  else float("nan")
    avg_smape    = sum(valid_smapes) / len(valid_smapes) if valid_smapes else float("nan")
    green_cases  = sum(1 for v in valid_mases if v < 0.5)
    ok_cases     = sum(1 for v in valid_mases if 0.5 <= v < 1.0)
    orange_cases = sum(1 for v in valid_mases if 1.0 <= v < 1.5)
    red_cases    = sum(1 for v in valid_mases if v >= 1.5)
    # Per-dataset avg RMSE (valid entries only)
    ds_avg_rmse: dict[str, float] = {}
    for ds in datasets:
        rmse_vals: list[float] = []
        for k in valid_keys:
            if k[0] == ds:
                r = data[k][1]
                if r is not None and not math.isnan(r):
                    rmse_vals.append(r)
        ds_avg_rmse[ds] = sum(rmse_vals) / len(rmse_vals) if rmse_vals else float("nan")
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
  &nbsp;·&nbsp; Primary metric: <strong>MASE</strong> (sMAPE for Dundee) &nbsp;·&nbsp;
  Secondary: <strong>sMAPE + RMSE</strong> &nbsp;·&nbsp;
  Best selection &amp; averages exclude IQR outliers
</p>
""")

    # Summary cards
    parts.append('<div class="summary-bar">')
    avg_mase_str  = f"{avg_mase:.3f}"  if not math.isnan(avg_mase)  else "—"
    avg_smape_str = f"{avg_smape:.1f}%" if not math.isnan(avg_smape) else "—"
    for num, lbl, col in [
        (total_cases,       "total cases",          "#2c3e50"),
        (n_discarded,       "discarded (outlier)",  "#c00000"),
        (avg_mase_str,      "avg MASE (valid)",     "#555"),
        (avg_smape_str,     "avg sMAPE (valid)",    "#555"),
        (green_cases,       "MASE < 0.5",           "#1a5c1a"),
        (ok_cases,          "MASE 0.5–1.0",         "#2e7d32"),
        (orange_cases,      "MASE 1.0–1.5",         "#7a4000"),
        (red_cases,         "MASE ≥ 1.5",           "#8b0000"),
    ]:
        parts.append(
            f'<div class="summary-card">'
            f'<div class="num" style="color:{col}">{num}</div>'
            f'<div class="lbl">{lbl}</div>'
            f'</div>'
        )
    parts.append('</div>')

    # Per-dataset avg RMSE row
    parts.append('<div class="summary-bar" style="margin-top:8px">')
    for ds in datasets:
        val = ds_avg_rmse.get(ds, float("nan"))
        val_str = f"{val:.1f}" if not math.isnan(val) else "—"
        parts.append(
            f'<div class="summary-card">'
            f'<div class="num" style="color:#2c3e50;font-size:1.2em">{val_str}</div>'
            f'<div class="lbl">avg RMSE — {ds} (valid)</div>'
            f'</div>'
        )
    parts.append('</div>')
    parts.append("""<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#b7e4b7"></div>MASE &lt; 0.5 (beats naïve 2×)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f0f8f0"></div>0.5 ≤ MASE &lt; 1.0 (better than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ffd580"></div>1.0 ≤ MASE &lt; 1.5 (worse than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f4b3b3"></div>MASE ≥ 1.5 (much worse than naïve)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e0e0e0"></div>No result</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#fff;outline:2px solid #c0932a;outline-offset:-2px"></div>★ Best per dataset &amp; horizon (non-discarded)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:repeating-linear-gradient(-45deg,rgba(180,0,0,.12),rgba(180,0,0,.12) 3px,transparent 3px,transparent 8px)"></div>⊘ Discarded — IQR outlier on ≥1 metric; excluded from averages &amp; best selection</div>
</div>
""")
    parts.append('</div>')  # close sticky-header

    # ── Best results summary table ─────────────────────────────────────────────
    scatter_store: list[dict] = []

    parts.append('<h2 style="margin:20px 0 8px">Best Results Summary</h2>')
    parts.append(
        '<table class="best-table">'
        '<thead><tr>'
        '<th>Horizon</th><th>Dataset</th><th>Strategy</th>'
        '<th>Best Model</th><th>Best FE</th>'
        '<th>MASE</th><th>sMAPE</th><th>RMSE</th>'
        '<th title="Min\u2013max of actual test values (populated on re-run)">y-range (test)</th>'
        '</tr></thead><tbody>'
    )
    prev_h: int | None = None
    for h in _HORIZONS:
        for ds in datasets:
            for ev in eval_strategies:
                combo = best_combo.get((ds, h, ev))
                if combo is None:
                    continue
                # ── Horizon separator row ─────────────────────────────────────
                if h != prev_h:
                    n_sep_cols = 9
                    parts.append(
                        f'<tr><td colspan="{n_sep_cols}" style="'
                        'background:#2c3e50;color:#fff;font-weight:700;'
                        'text-align:center;padding:4px 8px;letter-spacing:1px;'
                        f'font-size:11px">\u2500\u2500 {h} day{"s" if h > 1 else ""} \u2500\u2500</td></tr>'
                    )
                    prev_h = h
                best_mdl, best_fe = combo
                best_key = (ds, best_mdl, best_fe, ev, h)
                mase, rmse, smape = data.get(best_key, (None, None, None))
                mase_str  = f"{mase:.3f}"   if mase  is not None and not math.isnan(mase)  else "—"
                smape_str = f"{smape:.1f}%" if smape is not None and not math.isnan(smape) else "—"
                rmse_str  = f"{rmse:.1f}"   if rmse  is not None and not math.isnan(rmse)  else "—"
                bg    = _mase_bg(mase)
                tc    = _mase_text_color(mase)
                bg_sm = _smape_bg(smape)
                tc_sm = _smape_text_color(smape)
                yr = yrange.get((ds, h))
                yr_str = f"{yr[0]:.0f}\u2013{yr[1]:.0f}" if yr else "\u2014"

                # Build scatter points for this (h, ds, ev) group
                scatter_pts: list[dict] = []
                for mdl in models:
                    for fe in fe_cols:
                        k = (ds, mdl, fe, ev, h)
                        if k not in data:
                            continue
                        m, r, s = data[k]
                        if m is None or math.isnan(m) or s is None or math.isnan(s):
                            continue
                        scatter_pts.append({
                            "model": mdl,
                            "fe": fe if fe else "(baseline)",
                            "fe_raw": fe,
                            "mase": round(m, 4),
                            "smape": round(s, 4),
                            "rmse": round(r, 1) if r is not None and not math.isnan(r) else None,
                            "isBest": (mdl, fe) == combo,
                            "isDiscarded": k in discarded_set,
                        })

                sc_idx = len(scatter_store)
                scatter_store.append({
                    "title": f"{ds}  ·  {h}d  ·  {ev.upper()}",
                    "ds": ds,
                    "h": h,
                    "ev": ev,
                    "points": scatter_pts,
                })
                fe_label = best_fe if best_fe else "(baseline)"
                parts.append(
                    f'<tr class="best-row" data-scatter-id="{sc_idx}">'
                    f'<td>{h}d</td>'
                    f'<td>{ds}</td>'
                    f'<td>{ev.upper()}</td>'
                    f'<td class="best-model">{best_mdl}</td>'
                    f'<td>{fe_label}</td>'
                    f'<td class="metric-mase" style="background:{bg};color:{tc}">{mase_str}</td>'
                    f'<td style="background:{bg_sm};color:{tc_sm}">{smape_str}</td>'
                    f'<td>{rmse_str}</td>'
                    f'<td style="font-size:10px;color:#555;white-space:nowrap">{yr_str}</td>'
                    f'</tr>'
                )
    parts.append('</tbody></table>')

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
                            discarded = key in discarded_set
                            parts.append(_cell(mase, rmse, smape, ds, mdl, fe, h, ev, is_best=is_best, is_discarded=discarded))
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
      <button class="lb-tab" id="lb-tab-win" data-type="win">Window Sample</button>
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
      win:   td.dataset.win || '',
      title: td.dataset.title,
    };
  });

  const tabWin = document.getElementById('lb-tab-win');
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
    // Show/hide the Window Sample tab depending on whether this case has a win path
    const info = (imgData[caseId] || {})[h];
    const hasWin = info && info.win;
    tabWin.style.display = hasWin ? '' : 'none';
    // If current tab is 'win' but this case has no window sample, reset to 'avp'
    if (curType === 'win' && !hasWin) {
      curType = 'avp';
      document.querySelectorAll('.lb-tab').forEach(b => b.classList.remove('active'));
      document.querySelector('.lb-tab[data-type="avp"]').classList.add('active');
    }
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
  window._lbOpen = open;
})();
</script>
""")

    # ── Scatter modal ─────────────────────────────────────────────────────────
    parts.append("""
<div id="sc-overlay">
  <div id="sc-box">
    <div id="sc-header">
      <span id="sc-title">—</span>
      <span id="sc-subtitle">click a point to view prediction charts</span>
      <button id="sc-close" title="Close (Esc)">&#x2715;</button>
    </div>
    <div id="sc-plot"></div>
    <div id="sc-legend"></div>
  </div>
</div>
<div id="sc-tooltip"></div>
""")
    parts.append(
        '<script>window._scatterStore = '
        + json.dumps(scatter_store).replace('</script>', r'<\/script>')
        + ';</script>'
    )
    parts.append("""
<script>
(function () {
  'use strict';
  const MODEL_COLORS = {
    lightgbm: '#3498db', xgboost: '#9b59b6',
    lstm: '#e67e22', transformer: '#e74c3c', hybrid: '#2ecc71',
    dl_baseline_lstm: '#f39c12', dl_baseline_transformer: '#1abc9c',
    dl_baseline_hybrid: '#34495e',
  };

  /** One-letter abbreviation per FE component: L=log R=revin C=cal D=diff W=rolling */
  function feAbbr(feRaw) {
    if (!feRaw) return '\u2205';
    return feRaw.split('_').map(p =>
      p === 'log' ? 'L' : p === 'revin' ? 'R' : p === 'cal' ? 'C' :
      p === 'diff' ? 'D' : p === 'rolling' ? 'W' : '?'
    ).join('');
  }

  const scOverlay = document.getElementById('sc-overlay');
  const scClose   = document.getElementById('sc-close');
  const scTitle   = document.getElementById('sc-title');
  const scPlot    = document.getElementById('sc-plot');
  const scLegend  = document.getElementById('sc-legend');
  const scTooltip = document.getElementById('sc-tooltip');

  function niceTicks(mn, mx, n) {
    const range = mx - mn;
    if (range < 1e-10) return [mn];
    const rough = range / n;
    const exp = Math.pow(10, Math.floor(Math.log10(rough)));
    const frac = rough / exp;
    const step = (frac < 1.5 ? 1 : frac < 3.5 ? 2 : frac < 7.5 ? 5 : 10) * exp;
    const start = Math.ceil((mn - step * 1e-9) / step) * step;
    const tks = [];
    for (let i = 0; i < 30; i++) {
      const t = start + i * step;
      if (t > mx + step * 0.01) break;
      tks.push(parseFloat(t.toFixed(10)));
    }
    return tks;
  }

  function renderSVG(pts) {
    const W = 600, H = 430, ML = 68, MR = 28, MT = 32, MB = 54;
    const PW = W - ML - MR, PH = H - MT - MB;

    // Compute axis range from non-discarded points only so that extreme
    // outliers (marked as discarded) do not compress the visible area.
    const validPts = pts.filter(p => !p.isDiscarded);
    const axPts = validPts.length > 0 ? validPts : pts;
    const mases  = axPts.map(p => p.mase);
    const smapes = axPts.map(p => p.smape);
    let x0 = Math.min(...mases), x1 = Math.max(...mases);
    let y0 = Math.min(...smapes), y1 = Math.max(...smapes);
    const xp = (x1 - x0) * 0.14 || 0.2, yp = (y1 - y0) * 0.14 || 2;
    x0 -= xp; x1 += xp; y0 -= yp; y1 += yp;

    // Count discarded points that fall outside the axis range
    const clipped = pts.filter(p => p.isDiscarded && (p.mase < x0 || p.mase > x1 || p.smape < y0 || p.smape > y1));

    const xs = v => ML + (v - x0) / (x1 - x0) * PW;
    const ys = v => MT + PH - (v - y0) / (y1 - y0) * PH;
    const xt = niceTicks(x0, x1, 7), yt = niceTicks(y0, y1, 7);

    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-height:460px;display:block">`;

    // Plot area background
    s += `<rect x="${ML}" y="${MT}" width="${PW}" height="${PH}" fill="#fafafa" stroke="#ccc" stroke-width="0.5"/>`;

    // Grid lines
    s += '<g stroke="#e8e8e8" stroke-width="1">';
    for (const t of xt) s += `<line x1="${xs(t).toFixed(1)}" y1="${MT}" x2="${xs(t).toFixed(1)}" y2="${MT+PH}"/>`;
    for (const t of yt) s += `<line x1="${ML}" y1="${ys(t).toFixed(1)}" x2="${ML+PW}" y2="${ys(t).toFixed(1)}"/>`;
    s += '</g>';

    // MASE=1 naïve reference line
    if (x0 < 1 && x1 > 1) {
      const rx = xs(1).toFixed(1);
      s += `<line x1="${rx}" y1="${MT}" x2="${rx}" y2="${MT+PH}" stroke="#e74c3c" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.55"/>`;
      s += `<text x="${(xs(1)+4).toFixed(1)}" y="${MT+13}" font-size="9" fill="#e74c3c" opacity="0.7">MASE=1</text>`;
    }

    // Axes
    s += `<line x1="${ML}" y1="${MT}" x2="${ML}" y2="${MT+PH}" stroke="#555" stroke-width="1.5"/>`;
    s += `<line x1="${ML}" y1="${MT+PH}" x2="${ML+PW}" y2="${MT+PH}" stroke="#555" stroke-width="1.5"/>`;

    // X tick labels
    s += '<g font-size="10" fill="#555" text-anchor="middle">';
    for (const t of xt) s += `<text x="${xs(t).toFixed(1)}" y="${MT+PH+14}">${t.toFixed(2)}</text>`;
    s += '</g>';
    // Y tick labels
    s += '<g font-size="10" fill="#555" text-anchor="end">';
    for (const t of yt) s += `<text x="${ML-5}" y="${(ys(t)+3.5).toFixed(1)}">${t.toFixed(1)}</text>`;
    s += '</g>';

    // Axis labels
    s += `<text x="${ML+PW/2}" y="${H-8}" font-size="12" fill="#333" text-anchor="middle" font-weight="600">MASE</text>`;
    s += `<text x="11" y="${(MT+PH/2).toFixed(1)}" font-size="12" fill="#333" text-anchor="middle" font-weight="600" transform="rotate(-90,11,${(MT+PH/2).toFixed(1)})">sMAPE (%)</text>`;

    // Note about clipped extreme outliers
    if (clipped.length > 0) {
      const note = `${clipped.length} extreme outlier${clipped.length > 1 ? 's' : ''} outside chart range (axis set to non-discarded points)`;
      s += `<text x="${ML+PW/2}" y="${MT-8}" font-size="9" fill="#c00" text-anchor="middle" font-style="italic">${note}</text>`;
    }

    // Points — discard those outside the axis window; render discarded-but-visible last
    const inBounds = p => p.mase >= x0 && p.mase <= x1 && p.smape >= y0 && p.smape <= y1;
    const sorted = [...pts].filter(inBounds).sort((a, b) => (+b.isDiscarded - +a.isDiscarded) || (+a.isBest - +b.isBest));
    for (const pt of sorted) {
      const cx = xs(pt.mase).toFixed(1), cy = ys(pt.smape).toFixed(1);
      const col = MODEL_COLORS[pt.model] || '#888';
      const r   = pt.isBest ? 10 : 7;
      const fill   = pt.isDiscarded ? '#ddd' : col;
      const stroke = pt.isBest ? '#c0932a' : (pt.isDiscarded ? '#aaa' : 'rgba(255,255,255,0.7)');
      const sw     = pt.isBest ? 2.5 : 1.5;
      const op     = pt.isDiscarded ? 0.5 : 1.0;
      const cursor = pt.isDiscarded ? 'default' : 'pointer';
      const rmseStr = pt.rmse != null ? `   RMSE: ${pt.rmse}` : '';
      const flags   = (pt.isDiscarded ? ' [DISCARDED]' : '') + (pt.isBest ? ' \u2605 BEST' : '');
      const tip = `${pt.model} / ${pt.fe}\nMASE: ${pt.mase.toFixed(3)}   sMAPE: ${pt.smape.toFixed(1)}%${rmseStr}${flags}`;
      const cls   = pt.isDiscarded ? '' : 'sc-pt';
      const attrs = pt.isDiscarded ? '' : ` data-model="${pt.model}" data-fe-raw="${pt.fe_raw}"`;

      s += `<g class="${cls}" opacity="${op}" style="cursor:${cursor}"${attrs}>`;
      s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
      if (!pt.isDiscarded) {
        const abbr = feAbbr(pt.fe_raw);
        const fsz = pt.isBest ? 7 : 6;
        const ty  = pt.isBest ? (+cy - 1.5).toFixed(1) : (+cy + 2.5).toFixed(1);
        s += `<text x="${cx}" y="${ty}" text-anchor="middle" font-size="${fsz}" fill="rgba(255,255,255,0.95)" pointer-events="none" font-weight="700">${abbr}</text>`;
      }
      if (pt.isBest)
        s += `<text x="${cx}" y="${(+cy+5.5).toFixed(1)}" text-anchor="middle" font-size="11" fill="#c0932a" pointer-events="none">&#9733;</text>`;
      if (pt.isDiscarded) {
        const d = 4.5, cxn = +cx, cyn = +cy;
        s += `<line x1="${(cxn-d).toFixed(1)}" y1="${(cyn-d).toFixed(1)}" x2="${(cxn+d).toFixed(1)}" y2="${(cyn+d).toFixed(1)}" stroke="#c00" stroke-width="1.4" opacity="0.7" pointer-events="none"/>`;
        s += `<line x1="${(cxn+d).toFixed(1)}" y1="${(cyn-d).toFixed(1)}" x2="${(cxn-d).toFixed(1)}" y2="${(cyn+d).toFixed(1)}" stroke="#c00" stroke-width="1.4" opacity="0.7" pointer-events="none"/>`;
      }
      s += `<title>${tip}</title></g>`;
    }
    s += '</svg>';
    return s;
  }

  function buildLegend(pts) {
    const models = [...new Set(pts.map(p => p.model))];
    let h = '<div style="display:flex;flex-wrap:wrap;gap:12px;padding:8px 14px;font-size:11px;align-items:center">';
    for (const m of models) {
      const c = MODEL_COLORS[m] || '#888';
      h += `<span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:50%;background:${c};display:inline-block"></span>${m}</span>`;
    }
    h += '<span style="display:flex;align-items:center;gap:5px;margin-left:6px"><span style="width:10px;height:10px;border-radius:50%;background:#ddd;border:1px solid #aaa;display:inline-block"></span>discarded</span>';
    h += '<span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:50%;border:2.5px solid #c0932a;display:inline-block"></span>&#9733; best</span>';
    h += '</div>';
    h += '<div style="padding:2px 14px 8px;font-size:10px;color:#666">FE icons inside dots: <b>L</b>=log &nbsp; <b>R</b>=revin &nbsp; <b>C</b>=cal &nbsp; <b>D</b>=diff &nbsp; <b>W</b>=rolling &nbsp; <b>\u2205</b>=baseline (no FE)</div>';
    return h;
  }

  function openScatter(idx) {
    const entry = window._scatterStore[idx];
    scTitle.textContent = 'Results comparison for ' + entry.title;
    scPlot.innerHTML = renderSVG(entry.points);
    scLegend.innerHTML = buildLegend(entry.points);
    scPlot.dataset.ds = entry.ds;
    scPlot.dataset.h  = String(entry.h);
    scPlot.dataset.ev = entry.ev;
    scOverlay.classList.add('open');

    // Click on a valid point → close scatter and open image lightbox
    scPlot.querySelectorAll('.sc-pt').forEach(g => {
      g.addEventListener('click', e => {
        e.stopPropagation();
        const caseId = `${scPlot.dataset.ds}_${g.dataset.model}_${g.dataset.feRaw}_${scPlot.dataset.ev}`;
        scOverlay.classList.remove('open');
        if (window._lbOpen) window._lbOpen(caseId, parseInt(scPlot.dataset.h, 10));
      });
      // ── Tooltip ───────────────────────────────────────────────────────────
      g.addEventListener('mouseenter', () => {
        const titleEl = g.querySelector('title');
        if (titleEl) { scTooltip.textContent = titleEl.textContent; scTooltip.style.display = 'block'; }
      });
      g.addEventListener('mousemove', e => {
        const tx = e.clientX + 16, ty = e.clientY - 10;
        scTooltip.style.left = tx + 'px';
        scTooltip.style.top  = ty + 'px';
      });
      g.addEventListener('mouseleave', () => { scTooltip.style.display = 'none'; });
    });
  }

  document.querySelectorAll('.best-row').forEach(row => {
    row.addEventListener('click', () => openScatter(parseInt(row.dataset.scatterId, 10)));
  });
  scClose.addEventListener('click', () => { scOverlay.classList.remove('open'); scTooltip.style.display = 'none'; });
  scOverlay.addEventListener('click', e => { if (e.target === scOverlay) { scOverlay.classList.remove('open'); scTooltip.style.display = 'none'; } });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && scOverlay.classList.contains('open') &&
        !document.getElementById('lb-overlay').classList.contains('open'))
      scOverlay.classList.remove('open');
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
