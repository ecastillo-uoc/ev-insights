"""
Interface-agnostic model persistence.
Saves and loads trained models to/from the filesystem via models_dir.
Supports LightGBM (.ubj), XGBoost (.ubj), and Keras/LSTM (.keras + scaler).
"""
import logging
from datetime import datetime
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def save_model(forecaster_name: str, results: dict, models_dir: str, algo: str = None):
    """
    Save trained model(s) from results['train'] to models_dir.

    Args:
        forecaster_name: Name of the forecaster (for logging).
        results: Dict with 'train' key mapping model names to outputs.
        models_dir: Filesystem directory to store model files.
        algo: Algorithm name (for logging, not used for dispatch).
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    for model_name, output in results['train'].items():
        model_obj = output['model']
        train_params = output.get('params', {})

        if isinstance(model_obj, dict) and 'keras_model' in model_obj:
            _save_keras(model_obj, model_name, models_path, forecaster_name, train_params)
        else:
            _save_native(model_obj, model_name, models_path, forecaster_name)


def load_model(algo: str, model_name: str, models_dir: str):
    """
    Load a trained model from models_dir.

    Args:
        algo: Algorithm identifier ('lightgbm', 'xgboost', 'lstm').
        model_name: Name used when the model was saved.
        models_dir: Filesystem directory containing model files.

    Returns:
        The loaded model object.

    Raises:
        FileNotFoundError: If no matching model file exists.
    """
    models_path = Path(models_dir)

    # Try Keras/LSTM first
    keras_path = models_path / f"{model_name}.keras"
    if keras_path.exists():
        return _load_keras(model_name, models_path)

    # Try .ubj (LightGBM / XGBoost)
    ubj_path = models_path / f"{model_name}.ubj"
    if ubj_path.exists():
        return _load_native(algo, ubj_path)

    raise FileNotFoundError(
        f"No model file found for '{model_name}' in {models_path}. "
        f"Looked for: {keras_path.name}, {ubj_path.name}"
    )


def list_models(models_dir: str) -> list[str]:
    """
    List available model names in models_dir.

    Returns:
        Sorted list of unique model names (without extensions).
    """
    models = set()
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    for f in models_path.iterdir():
        if f.suffix in ('.ubj', '.keras'):
            models.add(f.stem)
    return sorted(models)


# --- Private helpers ---

def _save_keras(model_obj: dict, model_name: str, models_path: Path, forecaster_name: str, train_params: dict = None):
    """Save a Keras model, its scaler, and metadata to *models_path*."""
    keras_path = models_path / f"{model_name}.keras"
    scaler_path = models_path / f"{model_name}_scaler.pkl"
    meta_path = models_path / f"{model_name}_meta.pkl"
    model_obj['keras_model'].save(str(keras_path))
    joblib.dump(model_obj['scaler'], str(scaler_path))
    meta = {
        'look_back': model_obj.get('look_back', 30),
        'forecast_horizon': model_obj.get('forecast_horizon'),
        'params': train_params or {},
    }
    joblib.dump(meta, str(meta_path))
    logger.info(f"Saved LSTM model {forecaster_name} to {keras_path}")


def _save_native(model_obj, model_name: str, models_path: Path, forecaster_name: str):
    """Save a LightGBM/XGBoost model as ``.ubj``, rotating existing files."""
    model_file = models_path / f"{model_name}.ubj"
    if model_file.exists():
        ts = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        model_file.rename(models_path / f"{model_file.name}.{ts}")
    model_obj.save_model(str(model_file))
    logger.info(f"Saved forecast model {forecaster_name} to {model_file}")


def _load_keras(model_name: str, models_path: Path):
    """Load a Keras model, scaler, and metadata from *models_path*."""
    from keras.models import load_model as keras_load_model
    keras_path = models_path / f"{model_name}.keras"
    scaler_path = models_path / f"{model_name}_scaler.pkl"
    meta_path = models_path / f"{model_name}_meta.pkl"
    keras_model = keras_load_model(str(keras_path))
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    meta = joblib.load(str(meta_path)) if meta_path.exists() else {}
    logger.info(f"Loaded LSTM model from {keras_path}")
    return {
        'keras_model': keras_model,
        'scaler': scaler,
        'look_back': meta.get('look_back', 30),
        'forecast_horizon': meta.get('forecast_horizon'),
        'params': meta.get('params', {}),
    }


def _load_native(algo: str, ubj_path: Path):
    """Load a LightGBM or XGBoost model from a .ubj file.
    
    Tries the requested algo first, then falls back to the other one,
    since the file extension doesn't distinguish between the two formats.
    """
    loaders = []
    if algo == 'lightgbm':
        loaders = [('lightgbm', _load_lgb), ('xgboost', _load_xgb)]
    elif algo == 'xgboost':
        loaders = [('xgboost', _load_xgb), ('lightgbm', _load_lgb)]
    else:
        # Unknown algo: try both
        loaders = [('lightgbm', _load_lgb), ('xgboost', _load_xgb)]

    last_err = None
    for loader_algo, loader_fn in loaders:
        try:
            model = loader_fn(ubj_path)
            logger.info(f"Loaded {loader_algo} model from {ubj_path}")
            return model
        except Exception as e:
            last_err = e
            logger.debug(f"Failed to load {ubj_path} as {loader_algo}: {e}")

    raise ValueError(
        f"Could not load model from {ubj_path} (tried: {[a for a,_ in loaders]}). "
        f"Last error: {last_err}"
    )


def _load_lgb(ubj_path: Path):
    """Load a LightGBM Booster from a ``.ubj`` file."""
    import lightgbm as lgb
    return lgb.Booster(model_file=str(ubj_path))


def _load_xgb(ubj_path: Path):
    """Load an XGBoost model from a ``.ubj`` file."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(ubj_path))
    return model
