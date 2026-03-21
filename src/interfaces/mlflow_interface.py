import os
import requests
import time
from pathlib import Path
from pprint import pprint
from src.interfaces.interface import Interface
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException


# pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', 300)
# pd.set_option('display.width', 2000)


class MLflow(Interface):
    def __init__(self, name, type, output_dir, mlflow_dir, host, port, user, password, bucket_name, init_storage):
        super().__init__(name=name, type=type, output_dir=output_dir)

        self.mlflow_dir = Path(mlflow_dir)
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.bucket_name = bucket_name
        self.init_storage = init_storage
        self.tracking_uri = f"http://{self.host}:{self.port}"
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

        self.health_uri = self.tracking_uri + "/health"
        self.check_mlflow_health(self.health_uri)  # Perform the health check

        return

    def close_interface(self):
        return

    def get_current_run_id_from_model_name(self, model_name):
        # Retrieve all model versions
        model_versions = self.client.search_model_versions(f"name='{model_name}'")

        if not model_versions:
            raise Exception(f"No model found with the name {model_name}")

        # Assume we want the run_id of the latest model version
        latest_version = max(model_versions, key=lambda v: int(v.version))

        return latest_version.run_id

    def check_mlflow_health(self, url, timeout=30, interval=5):
        """
        Checks the health of mlflow service by sending an HTTP request to its health endpoint.

        Args:
        - url: The URL of the mlflow endpoint to check.
        - timeout: The maximum time in seconds to keep trying the check (default: 60).
        - interval: The time interval in seconds between each retry attempt (default: 5).

        Returns:
        - True if the service is reachable and responds correctly, False otherwise.
        """
        self.logger.info(f"Checking mlflow status: {url}")
        start_time = time.time()  # Record the start time

        while time.time() - start_time < timeout:  # Loop until the timeout is reached
            try:
                response = requests.get(url)  # Send a GET request to the health endpoint
                if response.status_code == 200:  # Check if the response status is 200 (OK)
                    self.logger.info("Mlflow is reachable and responding correctly.")
                    return True  # Return True if the service is healthy
                else:
                    raise RestException(f"Health check failed with status code: {response.status_code}")

            except requests.ConnectionError:  # Handle connection errors
                self.logger.error("Failed to connect to Mlflow. Retrying...")

            time.sleep(interval)  # Wait for the specified interval before retrying

        raise MlflowException("Timeout reached. Mlflow is not reachable.")

    def save_forecast_model(self, forecaster_name, algo, results, file_path=None):
        """
        # This function saves the forecast model
        """
        for model_name, output in results['train'].items():

            model = output['model']
            params = output['params']
            metrics = output['metrics']
            artifacts = output['artifacts']

            experiment_id = self.get_experiment_id(model_name)
            run_id = self.get_run_id(experiment_id)
            try:
                if params is not None and params != {}:
                    self.save_params(run_id=run_id, params=params)

                if model is not None and model != {}:
                    self.save_model(run_id=run_id, algo=algo, model=model, model_name=model_name)

                if metrics is not None and metrics != {}:
                    self.save_metrics(run_id=run_id, metrics=metrics)

                if artifacts is not None and artifacts != {}:
                    self.save_artifacts(run_id=run_id, artifacts=artifacts)
            finally:
                self.client.set_terminated(run_id)
        return

    def get_experiment_id(self, model_name):
        experiment = self.client.get_experiment_by_name(model_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(model_name)
        else:
            experiment_id = experiment.experiment_id
        return experiment_id

    def get_run_id(self, experiment_id):
        run = self.client.create_run(experiment_id)
        run_id = run.info.run_id
        return run_id

    def save_model(self, run_id, algo, model, model_name):
        self.logger.info(f"Saving model: run_id={run_id} algo={algo} model_name={model_name}")
        artifact_path = f"{algo}_model"

        # LSTM/Keras: model is a dict with keras_model, scaler, look_back
        if isinstance(model, dict) and 'keras_model' in model:
            import joblib, tempfile, os
            mlflow.keras.log_model(model['keras_model'], artifact_path=artifact_path,
                                   registered_model_name=model_name, run_id=run_id)
            # Save scaler and metadata as extra artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                if model.get('scaler') is not None:
                    scaler_path = os.path.join(tmpdir, 'scaler.pkl')
                    joblib.dump(model['scaler'], scaler_path)
                    self.client.log_artifact(run_id, scaler_path, artifact_path=artifact_path)
                meta = {'look_back': model.get('look_back', 30)}
                meta_path = os.path.join(tmpdir, 'meta.pkl')
                joblib.dump(meta, meta_path)
                self.client.log_artifact(run_id, meta_path, artifact_path=artifact_path)
        else:
            mlflow_module = self._get_mlflow_module(algo)
            mlflow_module.log_model(model, artifact_path=artifact_path,
                                    registered_model_name=model_name, run_id=run_id)
        return

    @staticmethod
    def _get_mlflow_module(algo):
        """Map algo name to the corresponding mlflow module."""
        mapping = {
            'xgboost': mlflow.xgboost,
            'lightgbm': mlflow.lightgbm,
            'keras': mlflow.keras,
            'lstm': mlflow.keras,
        }
        module = mapping.get(algo)
        if module is None:
            # Fallback: try getattr on mlflow
            module = getattr(mlflow, algo, None)
        if module is None:
            raise ValueError(f"No MLflow module found for algo '{algo}'")
        return module

    def save_params(self, run_id, params):
        self.logger.info(f"Saving params: run_id={run_id} params={params}")
        for param_name, param_value in params.items():
            self.client.log_param(run_id, param_name, param_value)
        return

    def save_metrics(self, run_id, metrics):
        self.logger.info(f"Saving metrics: run_id={run_id} params={metrics}")
        for metric_name, metric_value in metrics.items():
            self.client.log_metric(run_id, metric_name, metric_value)
        return

    def save_artifacts(self, run_id, artifacts):
        self.logger.info(f"Saving artifacts: run_id={run_id} params={artifacts}")
        for artifact_name, artifact_value in artifacts.items():
            self.client.log_artifact(run_id, artifact_name, artifact_value)
        return

    # TODO
    # def search_experiment(self, experiment_name):
    #     experiment = self.client.search_experiments(
    #         filter_string=f"tags.`project_name` = {experiment_name}"
    #     )
    #
    #     pprint(vars(experiment[0]))
    #
    #     if len(experiment) == 0:
    #         self.create_experiment(experiment_name=experiment_name)
    #     else:
    #         return

    # TODO
    def create_experiment(self, experiment_name):

        # Provide an Experiment description that will appear in the UI
        experiment_description = (
            "This is the grocery forecasting project. "
            "This experiment contains the produce models for apples."
        )

        # Provide searchable tags that define characteristics of the Runs that
        # will be in this Experiment
        experiment_tags = {
            "project_name": "grocery-forecasting",
            "store_dept": "produce",
            "team": "stores-ml",
            "project_quarter": "Q3-2023",
            "mlflow.note.content": experiment_description,
        }

        # Create the Experiment, providing a unique name
        produce_apples_experiment = self.client.create_experiment(
            name="Apple_Models", tags=experiment_tags
        )

        return

    def get_model(self, algo, model_name, models_dir=None):
        """
        Loads a machine learning model from an MLflow experiment.

        Args:
            algo (str): The algorithm name (e.g., "xgboost", "lightgbm", "lstm").
            model_name (str): The name of the model registered in MLflow.
            models_dir (str, optional): Not used for MLflow (present for interface compatibility).

        Returns:
            tuple: (experiment_id, run_id, model)
        """
        run_id = self.get_current_run_id_from_model_name(model_name=model_name)
        experiment = mlflow.get_experiment_by_name(model_name)
        artifact_path = f"{algo}_model"
        self.logger.info(f"Loading model_name: {model_name}, run_id: {run_id}, experiment_id: {experiment.experiment_id}")
        model_uri = f"mlflow-artifacts:/{experiment.experiment_id}/{run_id}/artifacts/{artifact_path}"

        # LSTM/Keras: load keras model + scaler + metadata
        if algo in ('lstm', 'keras'):
            import joblib, tempfile, os
            keras_model = mlflow.keras.load_model(model_uri=model_uri)
            # Download scaler and meta artifacts
            artifacts_dir = self.client.download_artifacts(run_id, artifact_path)
            scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
            meta_path = os.path.join(artifacts_dir, 'meta.pkl')
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
            model = {
                'keras_model': keras_model,
                'scaler': scaler,
                'look_back': meta.get('look_back', 30),
            }
        else:
            mlflow_module = self._get_mlflow_module(algo)
            model = mlflow_module.load_model(model_uri=model_uri)

        if model is not None:
            self.logger.info(f"Model loaded: {type(model)}")
        else:
            raise Exception("Model not found")

        return experiment.experiment_id, run_id, model

    # --- Stubs for Interface abstract methods not applicable to MLflow ---

    def get_data(self, data_selection: dict):
        raise NotImplementedError("MLflow interface does not support get_data")

    def init_db(self):
        pass  # MLflow manages its own DB schema

    def delete_dataset(self, dataset_name):
        raise NotImplementedError("MLflow interface does not support delete_dataset")

    def save_forecast_prediction(self, forecaster_name, actor, model, experiment_id, run_id, results, algo):
        raise NotImplementedError("MLflow interface does not support save_forecast_prediction")
