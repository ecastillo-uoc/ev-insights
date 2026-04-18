import json
import sys
import argparse
import logging
from shutil import copy
from datetime import datetime
from pathlib import Path, PureWindowsPath
from pprint import pprint
from src.utils.logger import Logger
from src.services.service import init_service
sys.path.append(str(Path(__file__).parent.parent))


def main(config_file=None, config_json=None, datasets_list=None, datasets_details_file=None):
    
    # Init logger
    Logger(config=config['utils']['logger'], filename="main")
    logger = logging.getLogger(__name__)
    logger.info("Start main")

    logger.debug(f"[main] main called. Service: {config_json.get('service') if config_json else 'None'}, datasets_list={datasets_list}")

    service = None
    output = []

    try:
        if config_file is None and config_json is None:
            raise Exception("Please provide config_file_path or config_json")
        elif config_file is not None and config_json is not None:
            raise Exception("Please only provide config_file_path or config_json")

        if config_file is not None:
            config_file_path = Path(config_file)
            config_json = json.loads(config_file_path.read_text().replace("\n", ""))
        elif config_json is not None:
            config_file_path = Path("api_" + config_json['service'])

        datetime_now = datetime.now()

        service_name = config_json['service']
        config = config_json['services'][service_name]

        # Override datasets_list and datasets_details_file from CLI arguments if provided
        if 'input' in config.get('interfaces', {}):
            if datasets_list is not None:
                logger.debug(f"[main] Overriding datasets_list with {datasets_list}")
                config['interfaces']['input']['datasets_list'] = datasets_list
            if datasets_details_file is not None:
                config['interfaces']['input']['datasets_details_file'] = datasets_details_file

        # Init output dir
        _base_output_dir = PureWindowsPath(config['utils']['output_dir']).as_posix()
        output_dir = Path(_base_output_dir).resolve() / (
            datetime_now.strftime("%Y-%m-%d_%H.%M.%S") + "__" +
            Path(__file__).stem + "__" +
            config_file_path.stem
        )
        config['utils']['output_dir'] = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Init config dir
        config_dir = output_dir / 'config'
        config['utils']['config_dir'] = str(config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)

        # Init log dir
        _base_logger_dir = PureWindowsPath(config['utils']['logger']['output_dir']).as_posix()
        log_dir = output_dir / _base_logger_dir
        config['utils']['logger']['output_dir'] = str(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Init interfaces dir
        _base_input_dir = PureWindowsPath(config['interfaces']['input']['output_dir']).as_posix()
        input_interface_dir = output_dir / _base_input_dir
        config['interfaces']['input']['output_dir'] = str(input_interface_dir)
        input_interface_dir.mkdir(parents=True, exist_ok=True)

        _base_output_interface_dir = PureWindowsPath(config['interfaces']['output']['output_dir']).as_posix()
        output_interface_dir = output_dir / _base_output_interface_dir
        config['interfaces']['output']['output_dir'] = str(output_interface_dir)
        output_interface_dir.mkdir(parents=True, exist_ok=True)

        if "mlflow" in config['interfaces']:
            mlflow_dir = Path(PureWindowsPath(config['utils']['input_dir']).as_posix()) / Path(PureWindowsPath(config['interfaces']['mlflow']['mlflow_dir']).as_posix())
            config['interfaces']['mlflow']['mlflow_dir'] = str(mlflow_dir.resolve())
            mlflow_dir.mkdir(parents=True, exist_ok=True)

        # Init service dir
        output_service_dir = output_dir / PureWindowsPath(config['output_dir']).as_posix()
        config['output_dir'] = str(output_service_dir)
        output_service_dir.mkdir(parents=True, exist_ok=True)



        if config_file is not None:
            # Copy the config file into the output folder
            copy(config_file_path, config_dir)
            # Save also utils config file with full path for any inconvenience
            filename = Path(config_file_path).stem + "_fullpaths.json"
            file_path = config_dir / filename
            with file_path.open('w') as json_file:
                json.dump(config, json_file, indent=4)

        elif config_json is not None:
            # Write config into the output folder
            file_path = config_dir / "api_conf_file.json"
            with file_path.open('w') as json_file:
                json.dump(config, json_file, indent=4)

        # Init service
        logger.debug(f"[main] Initializing service: {service_name}")
        if service_name == 'forecast':
            forecast_tasks = config.get('forecast', [])
            for i, task in enumerate(forecast_tasks):
                logger.debug(f"[main] Task {i}: name={task.get('name')}, algo={task.get('algo')}, "
                             f"prediction_target={task.get('prediction_target')}, model_strategy={task.get('model_strategy')}")
        service = init_service(config=config)

        # Run service
        output_tmp = service.run()
        output.append(output_tmp)

        # Close service (if needed)
        if service is not None:
            service.close_service()

    except Exception as e:
        logging.critical(e, exc_info=True)
        output_tmp = str(e)
        output.append(output_tmp)

    return output


if __name__ == '__main__':
    # Read conf file
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', help='configuration file', required=True)
    arg_parser.add_argument('--datasets_list', help='List of datasets to ingest', nargs='+', required=False)
    arg_parser.add_argument('--datasets_details_file', help='Path to datasets details file', required=False)

    # Set configuration
    args = arg_parser.parse_args()

    output = main(config_file=args.c, datasets_list=args.datasets_list, datasets_details_file=args.datasets_details_file)



