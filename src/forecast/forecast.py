import os
import copy
import logging
import importlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

from src.interfaces.interface import Interface
from src.utils.console import Colors


from .prediction_target_registry import (
    PredictionTarget, get_prediction_target_strategy,
    PREDICTION_TARGET_REGISTRY
)

from .model_registry import (
    ModelStrategyType, get_model_strategy
)
from src.utils.output import save_results_to_json

class Forecast(ABC):
    def __init__(self, id: int, name: str, algo: str, info: str, actor: str, actor_id: int, date: datetime.date, enabled: bool,
                 full_custom_mode: bool, mode: str, submode: str, models_dir: str, model_name: str, show_images: bool, save_images: bool,
                 save_results: bool, input_interface: Interface, output_interface: Interface, mlflow_interface: Interface, output_dir: str,
                 data_selection: dict, custom_params: dict):
        self.id = id
        self.name = name
        self.algo = algo
        self.info = info
        self.actor = actor
        self.actor_id = actor_id
        self.date = datetime.strptime(date, "%Y-%m-%d").date() if date is not  None else None
        self.enabled = enabled
        self.full_custom_mode = full_custom_mode
        self.mode = mode
        self.submode = submode
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.show_images = show_images
        self.save_images = save_images
        self.save_results = save_results
        self.input_interface = input_interface
        self.output_interface = output_interface
        self.mlflow_interface = mlflow_interface
        # TODO add merge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_selection = data_selection
        self.custom_params = custom_params
        self.logger = logging.getLogger('forecast')
        self.logger.info(f"Initialized {Colors.GREEN}{self.name}{Colors.NORMAL}")
        self.df = pd.DataFrame()
        self.model = None
        self.prediction = None
        self.datasets_names = []
        self.results = {}
        self.columns = []
        self.target = None
        return

    def load_data(self, df: pd.DataFrame, prediction=None, model=None):
        if df is not None:
            self.df = df
            self.datasets_names = self.df['dataset_name'].unique()
        if model is not None:
            self.model = model
        if prediction is not None:
            self.prediction = prediction

        return

    def check_data(self):
        # Remove NaT values from plug_in_datetime
        if self.df is not None:
            if 'plug_in_datetime' in self.df.columns:
                self.df = self.df.dropna(subset=['plug_in_datetime'])
        return

    @abstractmethod
    def feature_engineering(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def save_output_to_file(self):
        save_results_to_json(self.results, self.output_dir, self.name)

    def get_train_results(self, keys=None):
        results = copy.deepcopy(self.results)
        if keys:
            for key in self.results[self.mode]:
                for sub_key in self.results[self.mode][key]:
                    if sub_key not in keys:
                        results[self.mode][key].pop(sub_key)
        else:
            results = self.results

        return results

    def get_predict_results(self):
        return self.results

    @staticmethod
    def get_model_name(prefix, pilot=None, actor=None, actor_id=None):
        """
        Create a formatted string using the provided parameters.

        Parameters:
        - prefix (str): The required prefix for the string. Always included.
        - pilot (str or None): The pilot name. If None, it is excluded.
        - actor (str or None): The actor name, one of "u" (user), "c" (charging_station), "ug" (user group), "cg" (charging_station group).
            If None, it is excluded.
        - id (int or str or None): The optional identifier. If None, it is excluded.

        Returns:
        - str: A string in the format 'prefix_pilot_actor_id'

        Raises:
        - ValueError: If the actor is not in the set of valid values.
        """
        # Validate the actor parameter
        valid_actors = {"u", "c", "ug", "cg"}
        if actor is not None and actor not in valid_actors:
            raise ValueError(f"Invalid actor value: {actor}. Must be one of {valid_actors}.")

        components = [prefix]  # The prefix is always present

        if pilot:
            components.append(pilot)  # Add pilot if it's not None
        if actor:
            components.append(actor)  # Add actor if it's not None
        if actor_id:
            components.append(str(actor_id))  # Convert and add id if it's not None

        # Join the components with an underscore '_'
        model_name = "_".join(components)
        return model_name


def import_class(module_path, class_name):
    """
    Import a class by its string name.

    Args:
        module_path (str): The path of the module containing the class.
        class_name (str): The name of the class to import.

    Returns:
        The imported class.
    """
    module = importlib.import_module(module_path)
    _class = getattr(module, class_name)
    return _class


def init_forecast(config, models_dir, input_interface=None, output_interface=None, mlflow_interface=None):
    logger = logging.getLogger('init_forecast')
    forecast = None
    forecast_name = config["name"]
    
    config.setdefault('full_custom_mode', False)

    # --- Strategy override support ---
    # When the UI (or caller) provides 'prediction_target' and/or 'model_strategy',
    # we dynamically compose a GenericForecast instead of looking up a hardcoded class.
    prediction_target_key = config.get('prediction_target')
    model_strategy_key = config.get('model_strategy')

    logger.debug(f"[init_forecast] Config keys received: name={forecast_name}, algo={config.get('algo')}, "
                 f"prediction_target={prediction_target_key}, model_strategy={model_strategy_key}")
    logger.debug(f"[init_forecast] All config keys: {list(config.keys())}")

    if prediction_target_key or model_strategy_key:
        from .generic_forecast import GenericForecast


        # Resolve data strategy: from override or fall back to config name
        data_strategy = None
        if prediction_target_key:
            try:
                target_enum = PredictionTarget(prediction_target_key)
                data_strategy = get_prediction_target_strategy(target_enum)
                logger.info(f"Using prediction target override: {prediction_target_key}")
            except (ValueError, KeyError):
                logger.error(f"Unknown prediction_target: {prediction_target_key}")

        # Resolve model strategy: from override or fall back to config algo
        model_strategy = None
        resolved_algo = config.get('algo', 'unknown')
        if model_strategy_key:
            try:
                model_enum = ModelStrategyType(model_strategy_key)
                model_strategy = get_model_strategy(model_enum)
                resolved_algo = model_strategy_key
                logger.info(f"Using model strategy override: {model_strategy_key}")
            except (ValueError, KeyError):
                logger.error(f"Unknown model_strategy: {model_strategy_key}")

        # If only one override was given, resolve the other from the original config class
        if data_strategy and model_strategy:
            # Derive a meaningful name from the selections
            resolved_name = f"{resolved_algo}_{prediction_target_key}"
        elif data_strategy and not model_strategy:
            # Need model strategy from original class or default
            resolved_name = f"{config.get('algo', 'unknown')}_{prediction_target_key}"
            # Try to get the model strategy from the original forecast class
            try:
                import src.forecast.strategies as forecast_impl
                OrigClass = getattr(forecast_impl, forecast_name, None)
                if OrigClass and hasattr(OrigClass, '__init__'):
                    # Instantiate temporarily to extract strategy (GenericForecast stores it)
                    import inspect
                    sig = inspect.signature(OrigClass.__init__)
                    # The strategies are passed as positional args in __init__, read from source
                    # Safer: just use the algo from config to look up model strategy
                    algo_key = config.get('algo', 'xgboost')
                    model_enum = ModelStrategyType(algo_key)
                    model_strategy = get_model_strategy(model_enum)
            except Exception as e:
                logger.warning(f"Could not resolve model strategy from config algo: {e}")
        elif model_strategy and not data_strategy:
            resolved_name = f"{resolved_algo}_{forecast_name.split('_', 1)[1] if '_' in forecast_name else forecast_name}"
            # Try to get the data strategy from the original class
            try:
                import src.forecast.strategies as forecast_impl
                OrigClass = getattr(forecast_impl, forecast_name, None)
                if OrigClass:
                    # Create a temp instance to extract data_strategy - too complex
                    # Instead, parse the name convention: name contains the target hint
                    # e.g. xgboost_charge_duration -> session_duration, lightgbm_station_energy -> station_energy
                    name_parts = forecast_name.split('_', 1)
                    if len(name_parts) > 1:
                        target_hint = name_parts[1]
                        target_mapping = {
                            'charge_duration': PredictionTarget.SESSION_DURATION,
                            'energy': PredictionTarget.SESSION_ENERGY,
                            'station_charges': PredictionTarget.STATION_CHARGES,
                            'station_energy': PredictionTarget.STATION_ENERGY,
                        }
                        target_enum = target_mapping.get(target_hint)
                        if target_enum:
                            data_strategy = get_prediction_target_strategy(target_enum)
            except Exception as e:
                logger.warning(f"Could not resolve data strategy from config name: {e}")

        logger.debug(f"[init_forecast] Strategy resolution result: "
                     f"data_strategy={type(data_strategy).__name__ if data_strategy else None}, "
                     f"model_strategy={type(model_strategy).__name__ if model_strategy else None}")

        # --- Merge registry defaults into empty config fields ---
        if prediction_target_key:
            try:
                target_info = PREDICTION_TARGET_REGISTRY[PredictionTarget(prediction_target_key)]
                registry_defaults = target_info.default_params
            except (ValueError, KeyError):
                registry_defaults = {}

            # Merge default fields into data_selection if empty
            ds = config.get('data_selection', {})
            if not ds.get('fields'):
                default_fields = registry_defaults.get('fields', {})
                if default_fields:
                    ds['fields'] = dict(default_fields)  # copy
                    config['data_selection'] = ds
                    logger.debug(f"[init_forecast] Merged default fields from registry: {list(default_fields.keys())}")

            # Merge default custom_params if empty
            cp = config.get('custom_params', {})
            if not cp:
                default_cp = registry_defaults.get('custom_params', {})
                if default_cp:
                    config['custom_params'] = dict(default_cp)  # copy
                    logger.debug(f"[init_forecast] Merged default custom_params from registry: {list(default_cp.keys())}")

        if data_strategy and model_strategy and config.get("enabled", False):
            forecast = GenericForecast(
                data_strategy=data_strategy,
                model_strategy=model_strategy,
                id=config.get('id', 1),
                name=resolved_name,
                algo=resolved_algo,
                info=config.get('info', ''),
                actor=config.get('actor'),
                actor_id=config.get('actor_id'),
                date=config.get('date'),
                enabled=config['enabled'],
                full_custom_mode=config['full_custom_mode'],
                mode=config['mode'],
                submode=config.get('submode'),
                models_dir=str(Path(models_dir) / resolved_name),
                model_name=config['model_name'],
                show_images=config.get('show_images', False),
                save_images=config.get('save_images', False),
                save_results=config.get('save_results', True),
                input_interface=input_interface,
                output_interface=output_interface,
                mlflow_interface=mlflow_interface,
                output_dir=config['output_dir'],
                data_selection=config.get('data_selection') if not config.get('full_custom_mode') else None,
                custom_params=config.get('custom_params') if not config.get('full_custom_mode') else None,
            )
            logger.info(f"Created dynamic GenericForecast: name={resolved_name}, algo={resolved_algo}")
            return forecast
        elif not config.get("enabled", False):
            logger.info(f"Forecast {forecast_name} not enabled.")
            return None
        else:
            logger.error(f"Could not resolve both strategies from overrides. "
                         f"data_strategy={data_strategy}, model_strategy={model_strategy}. "
                         f"Falling back to config name lookup.")

    # --- Standard class lookup (no overrides or fallback) ---
    logger.debug(f"[init_forecast] Falling through to standard class lookup for: {forecast_name}")
    ForecastClass = None
    try:
        import src.forecast.strategies as forecast_impl
        if hasattr(forecast_impl, forecast_name):
             ForecastClass = getattr(forecast_impl, forecast_name)
    except ImportError:
        pass

    # Also check forecast_definitions (where the class combos live)
    if ForecastClass is None:
        try:
            import src.forecast.forecast_definitions as forecast_defs
            if hasattr(forecast_defs, forecast_name):
                ForecastClass = getattr(forecast_defs, forecast_name)
        except ImportError:
            pass
        
    # If not found, try legacy file import
    if ForecastClass is None and forecast_name in [p.stem for p in Path(__file__).parent.glob("*.py")]:
         try:
             full_module_path = f"src.forecast.{forecast_name}"
             ForecastClass = import_class(full_module_path, forecast_name)
         except Exception:
             pass

    if ForecastClass:
        if config["enabled"]:
            # Merge registry defaults for empty fields/custom_params
            # Infer prediction target from the class name: {algo}_{target}
            _algo_keys = ['lightgbm', 'xgboost', 'lstm']
            _inferred_target = None
            for _ak in _algo_keys:
                if forecast_name.startswith(_ak + '_'):
                    _inferred_target = forecast_name[len(_ak) + 1:]
                    break
            if _inferred_target:
                try:
                    _ti = PREDICTION_TARGET_REGISTRY[PredictionTarget(_inferred_target)]
                    _rd = _ti.default_params
                    ds = config.get('data_selection', {})
                    if not ds.get('fields'):
                        _df = _rd.get('fields', {})
                        if _df:
                            ds['fields'] = dict(_df)
                            config['data_selection'] = ds
                    cp = config.get('custom_params', {})
                    if not cp:
                        _dcp = _rd.get('custom_params', {})
                        if _dcp:
                            config['custom_params'] = dict(_dcp)
                except (ValueError, KeyError):
                    pass

            forecast = ForecastClass(id=config.get('id', 1),
                                     name=config['name'],
                                     algo=config['algo'],
                                     info=config['info'],
                                     actor=config.get('actor'),
                                     actor_id=config.get('actor_id'),
                                     date=config.get('date'),
                                     enabled=config['enabled'],
                                     full_custom_mode=config['full_custom_mode'],
                                     mode=config['mode'],
                                     submode=config.get('submode'),
                                     models_dir=str(Path(models_dir) / config['name']),
                                     model_name=config['model_name'],
                                     show_images=config['show_images'],
                                     save_images=config['save_images'],
                                     save_results=config['save_results'],
                                     input_interface=input_interface,
                                     output_interface=output_interface,
                                     mlflow_interface=mlflow_interface,
                                     output_dir=config['output_dir'],
                                     data_selection=config['data_selection'] if not config.get('full_custom_mode') else None,
                                     custom_params=config['custom_params'] if not config.get('full_custom_mode') else None)
        else:
            logger.info('Forecast ' + config["name"] + ' not enabled.')
            
    else:
         if config["enabled"]:
            # --- Name-based strategy inference fallback ---
            # Try to infer strategies from the forecast name pattern: {algo}_{target}
            # e.g. "lightgbm_session_energy" → model=lightgbm, target=session_energy
            logger.info(f"Class not found for '{forecast_name}', attempting name-based strategy inference")
            from .generic_forecast import GenericForecast as GenericForecast
            from .prediction_target_registry import (
                PredictionTarget as _PT,
                get_prediction_target_strategy as _get_pts,
                PREDICTION_TARGET_REGISTRY as _PTR,
            )
            from .model_registry import (
                ModelStrategyType as _MST,
                get_model_strategy as _get_ms,
            )

            _algo_map = {t.value: t for t in _MST}
            _target_map = {t.value: t for t in _PT}

            inferred_model = None
            inferred_data = None
            inferred_algo = None
            inferred_target_key = None

            # Try splitting name as "{algo}_{target}" with algo being one known token
            for algo_key in _algo_map:
                if forecast_name.startswith(algo_key + '_'):
                    remainder = forecast_name[len(algo_key) + 1:]
                    if remainder in _target_map:
                        inferred_algo = algo_key
                        inferred_target_key = remainder
                        inferred_model = _get_ms(_algo_map[algo_key])
                        inferred_data = _get_pts(_target_map[remainder])
                        break

            if inferred_model and inferred_data:
                logger.info(f"Inferred strategies from name '{forecast_name}': "
                            f"algo={inferred_algo}, target={inferred_target_key}")

                # Merge registry defaults for the inferred target
                try:
                    target_info = _PTR[_PT(inferred_target_key)]
                    registry_defaults = target_info.default_params
                except (ValueError, KeyError):
                    registry_defaults = {}

                ds = config.get('data_selection', {})
                if not ds.get('fields'):
                    default_fields = registry_defaults.get('fields', {})
                    if default_fields:
                        ds['fields'] = dict(default_fields)
                        config['data_selection'] = ds
                        logger.info(f"Merged default fields from name-based inference: {list(default_fields.keys())}")

                cp = config.get('custom_params', {})
                if not cp:
                    default_cp = registry_defaults.get('custom_params', {})
                    if default_cp:
                        config['custom_params'] = dict(default_cp)
                        logger.info(f"Merged default custom_params from name-based inference")

                forecast = GenericForecast(
                    data_strategy=inferred_data,
                    model_strategy=inferred_model,
                    id=config.get('id', 1),
                    name=forecast_name,
                    algo=inferred_algo,
                    info=config.get('info', ''),
                    actor=config.get('actor'),
                    actor_id=config.get('actor_id'),
                    date=config.get('date'),
                    enabled=config['enabled'],
                    full_custom_mode=config.get('full_custom_mode', False),
                    mode=config['mode'],
                    submode=config.get('submode'),
                    models_dir=str(Path(models_dir) / forecast_name),
                    model_name=config['model_name'],
                    show_images=config.get('show_images', False),
                    save_images=config.get('save_images', False),
                    save_results=config.get('save_results', True),
                    input_interface=input_interface,
                    output_interface=output_interface,
                    mlflow_interface=mlflow_interface,
                    output_dir=config['output_dir'],
                    data_selection=config.get('data_selection') if not config.get('full_custom_mode') else None,
                    custom_params=config.get('custom_params') if not config.get('full_custom_mode') else None,
                )
                logger.info(f"Created GenericForecast via name-based inference: {forecast_name}")
            else:

                # Retrieve the names of forecasts from the source, ensuring the list updates automatically whenever a new forecast is added.
                # New logic: check strategies package for subclasses of GenericForecast or Forecast explicitly
                try:
                    from . import strategies as forecast_impl
                    from .generic_forecast import GenericForecast
                    import inspect
                    
                    FORECAST = []
                    # Dynamic discovery from module
                    for name, obj in inspect.getmembers(forecast_impl):
                        if inspect.isclass(obj) and issubclass(obj, (Forecast, GenericForecast)) and obj is not Forecast and obj is not GenericForecast:
                            FORECAST.append(name)
                except ImportError:
                    # Fallback to file based if module not found (legacy support)
                    FORECAST = [p.stem for p in Path(__file__).parent.glob("*.py")
                            if p.name not in ["forecast.py", "_sample_forecast.py", "__init__.py", "generic_forecast.py"]]

                raise Exception(f'Forecast {config["name"]} does not exist or class not found in {FORECAST}')

    return forecast
