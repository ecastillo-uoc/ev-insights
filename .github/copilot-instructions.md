# EV-Insights — Copilot Instructions (Thesis Comparison Layer)

This file provides GitHub Copilot with the context needed to compare the
thesis document (`tfm/doc/vf/TFM_ECastillo.tex`) against the implementation
in `ev-insights/src/`.

**Primary use**: Ask Copilot to verify whether a thesis claim, algorithm
description, or parameter value matches the actual code.

---

## Thesis Section → Code Location Map

| Thesis section | Code location | Key files |
|---------------|--------------|-----------|
| **Data sources & preprocessing** | `src/data/` | `data_fetcher.py`, `constants.py` |
| **Database schema** | `src/sql/` | `schema.sql`, `sql_query.py` |
| **Feature engineering** | `src/forecast/feature_engineering.py` | transforms, calendar, lags |
| **Model architectures** | `src/forecast/strategies/` | one file per model |
| **Model architecture detail** | `src/forecast/strategies/.instructions.md` | exact layers + defaults |
| **Training pipeline** | `src/forecast/pipeline.py` | `run_forecast_pipeline()` |
| **Experiment configurations** | `src/tfm/tfm_forecast.py` | `execute_*`, `get_dataset_config()` |
| **Hussain et al. reproduction** | `src/forecast/strategies/dl_baseline_*.py` | article-faithful variants |
| **Results / metrics** | `output_metrics/forecast_metrics.csv` | per-model metric table |
| **Algorithm correctness** | `tfm/doc/algorithm_correctness_review.md` | formal audit |
| **Dundee tariff analysis** | `tfm/doc/dundee-analysis.md` | dataset-specific context |
| **System architecture** | `src/.instructions.md` | layered design, patterns |

---

## Terminology Mapping (Thesis ↔ Code)

| Thesis term | Code symbol / concept |
|-------------|----------------------|
| "Estrategia de datos" / data strategy | `PredictionTargetStrategy` in `strategies/interfaces.py` |
| "Estrategia de modelo" / model strategy | `ModelStrategy` in `strategies/interfaces.py` |
| "Horizonte de predicción" / forecast horizon | `forecast_horizon` param in `strategy_params` |
| "Ventana de contexto" / look-back window | `look_back` param; fixed at `LOOK_BACK_DAYS=28` for our models |
| "Diferenciación estacional de 7 días" | `use_differencing=True` in `apply_forward_transforms()` |
| "Transformación logarítmica" | `use_log_transform=True` → `np.log1p` / `np.expm1` |
| "Normalización por ventana" / RevIN | `use_window_norm=True`; `window_norm_days=28` |
| "Modelo Hussain LSTM" | `HussainTransformerModelStrategy` (transformer) OR `LSTMModelStrategy` with Hussain params |
| "Modelo Hussain Híbrido" | `HussainHybridModelStrategy` |
| "Backtest" / evaluación histórica | `_predict_backtest()` in `base_keras.py` |
| "Predicción recursiva" / schedule | `_predict_schedule()` in `base_keras.py` |
| "División COVID" / COVID split | `COVID_START='2020-08-05'`, `COVID_END='2020-11-17'` in `constants.py` |
| "Sesión de carga" | `ChargingSession` table; key col `energy_supplied` (kWh) |
| "Energia diaria por estación" | `fetch_daily_energy_for_forecast()` → `y` column (kWh/day) |
| RMSE, MAE, SMAPE, MASE | computed in `pipeline.py` on **original unscaled** values |

---

## Algorithm Implementation Notes (for thesis cross-check)

### LSTM
- Architecture: `LSTM(64, return_seq=True) → Drop → LSTM(32) → Drop → Dense(1)`
- Verified in `lstm_strategy.py:build_model()`

### Transformer
- N=2 blocks (default); each block: `LN → MHA → Add → LN → FF → Add`
- Head: `GAP → Dense(128) → Drop → Dense(1)`
- Verified in `transformer_strategy.py:build_model()`

### Hybrid LSTM-Transformer (ours)
- `Dense(d_model) + SinPE → LSTM(d_model) → [N=4 shared blocks: MHA → Add+LN → FFN → Add+LN] → GAP → Dense`
- **Universal Transformer weight tying** (Dehghani ICLR 2019) — MHA/FFN/LN weights shared across N=4 depth passes (zero extra params vs N=2)
- **No decoder** — encoder-only; decoder removed as it was circular (fed enc_out, attended back to enc_out)
- d_model=64, ff_dim=128, ~70 800 params (7× lighter than old ~520 k)
- Verified in `hybrid_strategy.py:build_model()`

### Hussain Transformer (article-faithful)
- `Dense(64,relu) → MHA(dropout=0.2) → GAP → Dense(1)`
- **No residuals, no LN, no FF block** — matches Fig. 3 of paper
- Verified in `dl_baseline_transformer_strategy.py:build_model()`

### Hussain Hybrid (article-faithful)
- Encoder+Decoder: `LSTM → SinPE → MHA → LN → Drop` (no residuals)
- Sinusoidal PE: Vaswani 2017 Eq. 3, added via `Lambda` layer
- Verified in `dl_baseline_hybrid_strategy.py:build_model()`

### Feature Engineering (forward/inverse order)
- Forward: `log1p → RevIN → diff(7)`  ← **order matters**
- Inverse: `undo_diff → undo_RevIN → expm1`
- TFM experiments use only `log1p` (differencing and RevIN disabled by default)
- Verified in `feature_engineering.py:apply_forward_transforms()`

---

## Known Justified Deviations from Referenced Papers

| Deviation | Justification | Location |
|-----------|--------------|----------|
| `look_back = 28` (fixed) for our models vs. Hussain's `look_back = h` | Decouples context from horizon; avoids anchoring predictions to stale regime | `tfm_forecast.py:LOOK_BACK_DAYS` |
| `look_back = max(14, h)` for Hussain variants (not exactly `h`) | Floor prevents very short context on small horizons | `tfm_forecast.py:MIN_LOOK_BACK` |
| Our Hybrid has residual connections; Hussain Hybrid does not | Hussain et al. variant exists for faithful reproduction; our variant follows Vaswani 2017 | `hybrid_strategy.py` vs. `dl_baseline_hybrid_strategy.py` |
| COVID excluded from our training; included in Hussain splits | Article methodology uses COVID data; ours avoids artificial zero contamination | `get_dataset_config(hussain=True/False)` |
| Paper "MSE" column ≈ numerically close to MAE → interpreted as RMSE | Mislabelling; RMSE is the correct comparison target | Note in `dl_baseline_transformer_strategy.py` docstring |
| `GlobalAveragePooling1D` in Hussain Transformer uses `channels_last` | Pools over temporal dimension (correct); `channels_first` would pool over features (wrong) | `dl_baseline_transformer_strategy.py` comment |

---

## How to Use This File for Thesis Review

1. **Verify an architecture claim**: Find the thesis statement, identify the thesis
   term, use the terminology map above to find the code symbol, then read the
   corresponding strategy file and `.instructions.md`.

2. **Verify a hyperparameter value**: Check `src/forecast/strategies/.instructions.md`
   for the default params table per model, and `src/tfm/tfm_forecast.py` for the
   actual `strategy_params` dict passed in `execute_*` functions.

3. **Verify a dataset split**: Check `src/tfm/tfm_forecast.py:get_dataset_config()`
   and the dataset table in `src/data/.instructions.md`.

4. **Verify transform pipeline**: Check `src/forecast/feature_engineering.py` and
   the "Target Transform Pipeline" section in `src/forecast/.instructions.md`.

5. **Verify metric computation**: Metrics are computed in `src/forecast/pipeline.py`
   after `apply_inverse_transforms()` — i.e., on original unscaled values.

---

## Instruction Files Index

| File | Coverage |
|------|----------|
| `src/.instructions.md` | Global architecture, patterns, conventions |
| `src/forecast/.instructions.md` | Forecast engine, pipeline, registries, transforms, metrics |
| `src/forecast/strategies/.instructions.md` | Per-model architectures + exact default params |
| `src/tfm/.instructions.md` | Experiment matrix, dataset configs, horizons, look-back constants |
| `src/data/.instructions.md` | Datasets, COVID constants, schema, fetcher functions |
| `src/interfaces/.instructions.md` | I/O backends (PostgreSQL, File, MLflow, ACN API) |
| `src/analysis/.instructions.md` | Analysis modules and auto-discovery |
| `src/services/.instructions.md` | Service orchestration layer |
| `src/sql/.instructions.md` | Database schema and query organization |
