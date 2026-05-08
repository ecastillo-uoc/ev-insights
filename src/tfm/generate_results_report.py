"""
generate_results_report.py
──────────────────────────
Read all metadata.json files from the experiment results directory and produce
an HTML report with one table per horizon.

Rows    = dataset × model
Columns = FE-tag combinations present in the data
Cell    = sMAPE  (primary metric; also shows RMSE as sub-text)

Colour coding
  sMAPE < 10   → green
  sMAPE < 30   → white (acceptable)
  sMAPE < 50   → orange
  sMAPE >= 50  → red
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
_DATASET_ORDER = ["ACN_Caltech", "Dundee", "ACN_JPL"]
_MODEL_ORDER = [
    "lightgbm", "xgboost",
    "lstm", "transformer", "hybrid",
    "hussain_lstm", "hussain_transformer", "hussain_hybrid",
]
_FE_ORDER = [
    "",
    "log", "log_cal",
    "revin", "log_revin", "log_revin_cal",
    "diff", "diff_cal",
    "log_diff", "log_diff_cal",
    "log_rolling", "log_diff_cal_rolling",
]
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


# ── Image path helper ─────────────────────────────────────────────────────────

def _img_paths(ds: str, mdl: str, fe: str, h: int) -> tuple[str, str, str]:
    """Return (avp, tts, zoom) paths relative to the HTML output file."""
    case_id = f"{ds}_{mdl}_{fe}"
    base    = f"chapters/results/{case_id}/{ds}"
    suffix  = f"{mdl}_{h}d_{ds}_{h}days"
    return (
        f"{base}_actual_vs_predict_{suffix}.png",
        f"{base}_train_test_split_{suffix}.png",
        f"{base}_train_test_split_{suffix}_zoom.png",
    )


# ── Colour helpers ────────────────────────────────────────────────────────────

def _smape_bg(smape: float | None) -> str:
    if smape is None or math.isnan(smape):
        return "#e0e0e0"   # grey — missing
    if smape < 10:
        return "#b7e4b7"   # green
    if smape < 30:
        return "#f0f8f0"   # very light green / neutral
    if smape < 50:
        return "#ffd580"   # orange
    return "#f4b3b3"       # red


def _smape_text_color(smape: float | None) -> str:
    if smape is None or math.isnan(smape):
        return "#777"
    if smape >= 50:
        return "#8b0000"
    if smape >= 30:
        return "#7a4000"
    if smape < 10:
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

.legend { display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
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

.summary-bar { display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap; }
.summary-card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
                padding: 10px 16px; text-align: center; min-width: 100px; }
.summary-card .num { font-size: 1.6em; font-weight: 700; }
.summary-card .lbl { font-size: 0.8em; color: #666; }

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


def _cell(smape: float | None, rmse: float | None,
          ds: str = "", mdl: str = "", fe: str = "", h: int = 0) -> str:
    bg = _smape_bg(smape)
    tc = _smape_text_color(smape)
    if smape is None or math.isnan(smape):
        return f'<td style="background:{bg}"><span class="no-data">—</span></td>'
    rmse_str = f"{rmse:.1f}" if rmse is not None and not math.isnan(rmse) else "—"
    avp, tts, zoom = _img_paths(ds, mdl, fe, h) if ds else ("", "", "")
    title = f"{ds} / {mdl} / {fe or '(baseline)'} — {h}d"
    data = (
        f' class="clickable"'
        f' data-case="{ds}_{mdl}_{fe}"'
        f' data-horizon="{h}"'
        f' data-avp="{avp}"'
        f' data-tts="{tts}"'
        f' data-zoom="{zoom}"'
        f' data-title="{title}"'
    )
    return (
        f'<td{data} style="background:{bg};color:{tc}">'
        f'<span class="metric-main">{smape:.1f}%</span>'
        f'<span class="metric-sub">RMSE {rmse_str}</span>'
        f'</td>'
    )


def build_html(records: list[dict]) -> str:
    # ── Index data: (dataset, model, fe_tag, horizon) → (smape, rmse)
    data: dict[tuple, tuple[float | None, float | None]] = {}
    fe_tags_seen: set[str] = set()
    datasets_seen: set[str] = set()
    models_seen: set[str] = set()

    for rec in records:
        ds  = rec.get("dataset", "")
        mdl = rec.get("model", "")
        fe  = rec.get("fe_tag", "")
        datasets_seen.add(ds)
        models_seen.add(mdl)
        fe_tags_seen.add(fe)
        metrics = rec.get("metrics", {})
        for h_key, m in metrics.items():
            h = int(h_key.rstrip("d"))
            smape = m.get("SMAPE")
            rmse  = m.get("RMSE")
            data[(ds, mdl, fe, h)] = (smape, rmse)

    # Ordered columns (only those with at least one result)
    fe_cols = [f for f in _FE_ORDER if f in fe_tags_seen]
    # Append any unseen tags not in the canonical order
    fe_cols += sorted(fe_tags_seen - set(_FE_ORDER))

    datasets = [d for d in _DATASET_ORDER if d in datasets_seen] + \
               sorted(datasets_seen - set(_DATASET_ORDER))
    models   = [m for m in _MODEL_ORDER if m in models_seen] + \
               sorted(models_seen - set(_MODEL_ORDER))

    # ── Summary stats
    total_cases   = len(records)
    green_cases   = sum(1 for (_, _, _, _), (s, _) in data.items() if s is not None and s < 10)
    orange_cases  = sum(1 for (_, _, _, _), (s, _) in data.items() if s is not None and 30 <= s < 50)
    red_cases     = sum(1 for (_, _, _, _), (s, _) in data.items() if s is not None and s >= 50)
    valid_smapes  = [s for (s, _) in data.values() if s is not None and not math.isnan(s)]
    avg_smape     = sum(valid_smapes) / len(valid_smapes) if valid_smapes else float("nan")

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
<h1>EV-Insights — Forecast Results Summary</h1>
<p class="subtitle">
  {total_cases} experiment cases &nbsp;·&nbsp;
  {len(datasets)} datasets &nbsp;·&nbsp;
  {len(models)} models &nbsp;·&nbsp;
  {len(fe_cols)} FE variants &nbsp;·&nbsp;
  {len(_HORIZONS)} horizons
  &nbsp;·&nbsp; Primary metric: <strong>sMAPE</strong> &nbsp;·&nbsp; Secondary: <strong>RMSE</strong>
</p>
""")

    # Summary cards
    parts.append('<div class="summary-bar">')
    for num, lbl, col in [
        (total_cases,  "total cases",   "#2c3e50"),
        (f"{avg_smape:.1f}%", "avg sMAPE", "#555"),
        (green_cases,  "sMAPE < 10%",   "#1a5c1a"),
        (orange_cases, "sMAPE 30–50%",  "#7a4000"),
        (red_cases,    "sMAPE ≥ 50%",   "#8b0000"),
    ]:
        parts.append(
            f'<div class="summary-card">'
            f'<div class="num" style="color:{col}">{num}</div>'
            f'<div class="lbl">{lbl}</div>'
            f'</div>'
        )
    parts.append('</div>')

    # Legend
    parts.append("""<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#b7e4b7"></div>sMAPE &lt; 10%</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f0f8f0"></div>10% ≤ sMAPE &lt; 30%</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ffd580"></div>30% ≤ sMAPE &lt; 50%</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f4b3b3"></div>sMAPE ≥ 50%</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#e0e0e0"></div>No result</div>
</div>
""")

    # One section per horizon
    for h in _HORIZONS:
        parts.append(f'<div class="horizon-section">')
        parts.append(f'<h2>Horizon: {h} day{"s" if h > 1 else ""}</h2>')
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
                # Only show model row if it has at least one result for this dataset
                has_any = any(
                    (ds, mdl, fe, h) in data for fe in fe_cols
                )
                if not has_any:
                    continue

                parts.append('<tr>')
                parts.append(f'<td class="row-header">&nbsp;&nbsp;{mdl}</td>')
                for fe in fe_cols:
                    key = (ds, mdl, fe, h)
                    if key in data:
                        smape, rmse = data[key]
                        parts.append(_cell(smape, rmse, ds, mdl, fe, h))
                    else:
                        parts.append('<td class="fe-col-missing"><span class="no-data">—</span></td>')
                parts.append('</tr>')

        parts.append('</tbody></table></div>')

    # FE legend table
    parts.append('<h2>FE Variant Descriptions</h2>')
    parts.append('<table style="margin-top:6px">')
    parts.append('<thead><tr><th>Tag</th><th>log</th><th>diff</th><th>calendar</th><th>RevIN</th><th>rolling</th></tr></thead><tbody>')
    fe_meta = {
        "":                    (False, False, False, False, False),
        "log":                 (True,  False, False, False, False),
        "log_cal":             (True,  False, True,  False, False),
        "diff":                (False, True,  False, False, False),
        "diff_cal":            (False, True,  True,  False, False),
        "log_diff":            (True,  True,  False, False, False),
        "log_diff_cal":        (True,  True,  True,  False, False),
        "revin":               (False, False, False, True,  False),
        "log_revin":           (True,  False, False, True,  False),
        "log_revin_cal":       (True,  False, True,  True,  False),
        "log_rolling":         (True,  False, False, False, True),
        "log_diff_cal_rolling":(True,  True,  True,  False, True),
    }
    for fe in fe_cols:
        meta = fe_meta.get(fe, (None, None, None, None, None))
        def _yn(v):
            if v is None: return "?"
            return "✓" if v else "·"
        label = _fe_label(fe)
        parts.append(
            f'<tr><td style="text-align:left;font-weight:600">{label}</td>'
            + "".join(f'<td>{_yn(v)}</td>' for v in meta)
            + '</tr>'
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
