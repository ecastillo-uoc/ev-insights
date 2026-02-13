import os
import copy
import logging
import importlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from pprint import pprint
from abc import abstractmethod
from src.interfaces.interface import Interface

# Retrieve the names of forecasts from the source, ensuring the list updates automatically whenever a new forecast is added.
# New logic: check forecast_implementations.py for subclasses of GenericForecast or Forecast explicitly
try:
    import src.forecast.forecast_implementations as forecast_impl
    from src.forecast.generic_forecast import GenericForecast
    import inspect
    
    FORECAST = []
    # Dynamic discovery from module
    for name, obj in inspect.getmembers(forecast_impl):
        if inspect.isclass(obj) and issubclass(obj, (Forecast, GenericForecast)) and obj is not Forecast and obj is not GenericForecast:
             FORECAST.append(name)
except ImportError:
    # Fallback to file based if module not found (legacy support)
    FORECAST = [p.stem for p in Path(__file__).parent.glob("*.py")
            if p.name not in ["forecast.py", "_sample_forecast.py", "__init__.py", "generic_forecast.py", "forecast_implementations.py"]]


class Forecast:
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
        self.logger.info("Initialized " + self.name)
        self.df = pd.DataFrame()
        self.model = None
        self.prediction = None
        self.datasets_names = []
        self.results = {}
        self.columns = []
        self.target = None
        return

    @abstractmethod
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

    # TODO This should be moved into the interface
    def save_output_to_file(self):
        # Save output to file
        filename = self.output_dir / f"{self.name}.json"
        with filename.open('w') as file:
            pprint(self.results, stream=file)

        return

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
    
    # Refresh FORECAST list if needed or rely on what was loaded at module level.
    # We should support both: specific module file (old way) and forecast_implementations (new way)
    
    # Check if name is in FORECAST (which now includes classes from forecast_implementations)
    # The check below might fail if FORECAST only has classes but not files if import failed? 
    # But we modified FORECAST to contain class names.
    
    # Try importing from forecast_implementations first
    ForecastClass = None
    try:
        import src.forecast.forecast_implementations as forecast_impl
        if hasattr(forecast_impl, forecast_name):
             ForecastClass = getattr(forecast_impl, forecast_name)
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
            if 'full_custom_mode' not in config.keys():
                config['full_custom_mode'] = False

            forecast = ForecastClass(id=config['id'] if 'id' in config.keys() else 1,
                                     name=config['name'],
                                     algo=config['algo'],
                                     info=config['info'],
                                     actor=config['actor'] if 'actor' in config.keys() else None,
                                     actor_id=config['actor_id'] if 'actor_id' in config.keys() else None,
                                     date=config['date'] if 'date' in config.keys() else None,
                                     enabled=config['enabled'],
                                     full_custom_mode=config['full_custom_mode'],
                                     mode=config['mode'],
                                     submode=config['submode'] if 'submode' in config.keys() else None,
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
         # Fallback error
         if config["enabled"]: # Only error if enabled? Original code raised exception regardless if not in FORECAST list?
             # Original: if config["name"] in FORECAST: ... else raise Exception
             raise Exception('Forecast ' + config["name"] + ' does not exist or class not found.')

    return forecast
