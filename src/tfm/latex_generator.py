"""
LaTeX fragment generator for per-case forecast results.

Called by :func:`~src.tfm.tfm_forecast.run_single_case` after each experiment
case completes.  Produces a self-contained ``.tex`` snippet that can be
``\\input``-ed into ``04_resultados.tex``.

Each fragment contains:

* A ``\\subsection`` heading with model and FE variant information.
* A configuration paragraph (feature-engineering flags, split date).
* A metrics table (all requested horizons × MSE / RMSE / MAE / MAPE / SMAPE).
* One ``\\includegraphics`` figure per horizon (actual vs predicted).
* A health-status note when any health flag is ``False``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any


# ── Display-name mappings ──────────────────────────────────────────────────

_MODEL_DISPLAY = {
    'lstm':                  'LSTM',
    'transformer':           'Transformer',
    'hybrid':                'Hybrid (LSTM+Transformer)',
    'lightgbm':              'LightGBM',
    'xgboost':               'XGBoost',
    'hussain_lstm':          'LSTM (Hussain et al.)',
    'hussain_transformer':   'Transformer (Hussain et al.)',
    'hussain_hybrid':        'Hybrid (Hussain et al.)',
}

_FE_TAG_DISPLAY = {
    'log':      'Log-transform',
    'log_cal':  'Log + Calendario',
    'log_diff': 'Log + Diferenciación',
}

_DATASET_DISPLAY = {
    'Dundee':      'Dundee',
    'ACN_Caltech': 'ACN Caltech',
    'ACN_JPL':     'ACN JPL',
    'ACN_Office001': 'ACN Office001',
    'BeLib':       'BeLib',
    'AMB_Barcelona': 'AMB Barcelona',
}


def _fmt(val: Any, decimals: int = 2) -> str:
    """Format a numeric value for LaTeX tables, or '---' if missing."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r'\textit{---}'
    if isinstance(val, float):
        return f'{val:,.{decimals}f}'
    return str(val)


def _escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return (
        text.replace('_', r'\_')
            .replace('%', r'\%')
            .replace('&', r'\&')
            .replace('#', r'\#')
    )


def generate_case_latex(
    case_id: str,
    metadata: dict,
    case_dir: Path,
) -> str:
    """Return a LaTeX fragment string for one experiment case.

    Parameters
    ----------
    case_id : str
        Unique identifier of the case (e.g. ``'Dundee_lstm_log'``).
    metadata : dict
        Metadata dict as written to ``metadata.json`` by
        :func:`~src.tfm.tfm_forecast.run_single_case`.
    case_dir : Path
        Directory where case artifacts live.  Used to determine the
        ``\\includegraphics`` paths relative to ``doc/vf/``.
    """
    model = metadata.get('model', case_id)
    dataset = metadata.get('dataset', '')
    fe_tag = metadata.get('fe_tag', '')
    fe_config = metadata.get('fe_config', {})
    split_date = metadata.get('split_date', '---')
    horizons = metadata.get('horizons', [])
    metrics_by_horizon: Dict[str, dict] = metadata.get('metrics', {})
    health = metadata.get('health', {})
    hussain = metadata.get('hussain_variant', False)

    model_display = _MODEL_DISPLAY.get(model, model)
    fe_display = _FE_TAG_DISPLAY.get(fe_tag, fe_tag)
    dataset_display = _DATASET_DISPLAY.get(dataset, dataset)

    # LaTeX label must not contain backslash or special chars
    label = case_id.replace('_', '-')

    # ── Path prefix relative to doc/vf/ where latexmk runs ────────────────
    # case_dir is Something like .../tfm/doc/vf/chapters/results/Dundee_lstm_log
    # \includegraphics resolves relative to doc/vf/ (the latexmk working dir)
    # So we build a path like: chapters/results/Dundee_lstm_log/<file>
    try:
        doc_vf_root = case_dir.parents[2]  # .../tfm/doc/vf
        img_prefix = str(case_dir.relative_to(doc_vf_root)).replace('\\', '/')
    except Exception:
        img_prefix = f'chapters/results/{case_id}'

    # ── Feature engineering summary ────────────────────────────────────────
    fe_items = []
    if fe_config.get('use_log_transform'):
        fe_items.append('log1p/expm1 en el objetivo')
    if fe_config.get('use_differencing'):
        fe_items.append('diferenciación de primer orden')
    if fe_config.get('use_calendar_features'):
        fe_items.append('características calendario (día semana, mes, sin/cos, día hábil)')
    fe_summary = '; '.join(fe_items) if fe_items else 'ninguna transformación adicional'
    hussain_note = r' El \textit{look\_back} es igual al horizonte de predicción (variante Hussain et al.).' if hussain else ''

    # ── Health status ──────────────────────────────────────────────────────
    health_warnings = []
    if not health.get('smape_ok', True):
        health_warnings.append(
            r'sMAPE $\geq 30\%$ en al menos un horizonte — revisar transformaciones inversas o calidad de datos.'
        )
    if not health.get('all_horizons_completed', True):
        missing = health.get('horizons_missing', [])
        health_warnings.append(
            rf'Horizontes no completados: {", ".join(str(h) + "d" for h in missing)}.'
        )

    # ── Build metrics table rows ───────────────────────────────────────────
    table_rows = []
    for h in horizons:
        key = f'{h}d'
        m = metrics_by_horizon.get(key, {})
        mse = _fmt(m.get('MSE'))
        rmse = _fmt(m.get('RMSE'))
        mae = _fmt(m.get('MAE'))
        mape = _fmt(m.get('MAPE') * 100 if m.get('MAPE') is not None else None)
        smape = _fmt(m.get('SMAPE'))
        table_rows.append(
            rf'        {h:>3}d & {mse} & {rmse} & {mae} & {mape} & {smape} \\'
        )
    table_body = '\n'.join(table_rows) if table_rows else r'        \multicolumn{6}{c}{\textit{Sin resultados}} \\'

    # ── Build figure blocks (one per horizon) ──────────────────────────────
    figure_blocks = []
    for h in horizons:
        # Filename pattern matches what pipeline.py generates
        # e.g. Dundee_actual_vs_predict_lstm_1d_Dundee_1days.png
        model_name_prefix = f'{model}_{h}d'
        model_name = f'{model_name_prefix}_{dataset}'
        fig_file = f'{dataset}_actual_vs_predict_{model_name}_{h}days.png'
        fig_path = f'{img_prefix}/{fig_file}'
        fig_label = f'fig:{label}-{h}d-avp'
        fig_caption = (
            rf'{dataset_display} — {_escape(model_display)}, '
            rf'{_escape(fe_display)}, horizonte {h} días: energía diaria (kWh) '
            rf'real vs.\ predicha. Los primeros \texttt{{look\_back}} días del '
            rf'conjunto de test se excluyen (calentamiento del modo \textit{{backtest}}).'
        )
        figure_blocks.append(
            rf"""
\begin{{figure}}[!htbp]
    \centering
    \IfFileExists{{{fig_path}}}{{\includegraphics[width=\textwidth]{{{fig_path}}}}}{{\fbox{{\textit{{Figura pendiente: {_escape(fig_file)}}}}}}}
    \caption{{{fig_caption}}}
    \label{{{fig_label}}}
\end{{figure}}
\FloatBarrier"""
        )

    figures_section = '\n'.join(figure_blocks)

    # ── Health note block ──────────────────────────────────────────────────
    if health_warnings:
        health_block = (
            r'\begin{tcolorbox}[colback=yellow!10!white,colframe=orange!80!black,'
            r'title=Advertencias de salud del caso]' + '\n'
            r'\begin{itemize}' + '\n'
            + '\n'.join(rf'\item {w}' for w in health_warnings) + '\n'
            r'\end{itemize}' + '\n'
            r'\end{tcolorbox}'
        )
    else:
        health_block = r'\textit{Estado: sin advertencias de salud.}'

    # ── Assemble the fragment ──────────────────────────────────────────────
    lines = [
        f'% Auto-generated case file — do not edit manually',
        f'% Case: {case_id}   Generated: {metadata.get("timestamp", "unknown")}',
        '',
        rf'\subsection{{{_escape(model_display)} --- {_escape(fe_display)}}}\label{{subsec:{label}}}',
        '',
        rf'\paragraph{{Configuración}} {fe_summary}.{hussain_note}',
        rf'Fecha de corte train/test: \texttt{{{split_date}}}.',
        rf'Horizontes evaluados: {", ".join(str(h) + "d" for h in horizons)}.',
        '',
        health_block,
        '',
        r'\paragraph{Métricas de error}',
        '',
        rf'\begin{{table}}[htbp]',
        rf'    \centering',
        rf'    \footnotesize',
        rf'    \begin{{tabular}}{{rrrrrr}}',
        rf'        \hline',
        rf'        \textbf{{H}} & \textbf{{MSE}} & \textbf{{RMSE}} & \textbf{{MAE}} & \textbf{{MAPE (\%)}} & \textbf{{SMAPE (\%)}} \\',
        rf'        \hline',
        table_body,
        rf'        \hline',
        rf'    \end{{tabular}}',
        rf'    \caption{{Métricas de predicción para {_escape(dataset_display)}, '
        rf'modelo {_escape(model_display)}, configuración FE: {_escape(fe_display)}.}}',
        rf'    \label{{tab:{label}-metrics}}',
        rf'\end{{table}}',
        '',
        r'\FloatBarrier',
        '',
        r'\paragraph{Predicciones por horizonte}',
        figures_section,
    ]

    return '\n'.join(lines) + '\n'
