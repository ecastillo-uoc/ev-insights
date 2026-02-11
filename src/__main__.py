import json
import os
import sys
import argparse
import logging
from shutil import copy
from datetime import datetime
from pathlib import Path
from pprint import pprint
from src.utils.logger import Logger
from src.services.service import init_service
sys.path.append(str(Path(__file__).parent.parent))


def main(config_file=None, config_json=None):

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

        # Init output dir
        output_dir = Path(config['utils']['output_dir']).resolve() / (
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
        log_dir = output_dir / config['utils']['logger']['output_dir']
        config['utils']['logger']['output_dir'] = str(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Init interfaces dir
        input_interface_dir = output_dir / config['interfaces']['input']['output_dir']
        config['interfaces']['input']['output_dir'] = str(input_interface_dir)
        input_interface_dir.mkdir(parents=True, exist_ok=True)

        output_interface_dir = output_dir / config['interfaces']['output']['output_dir']
        config['interfaces']['output']['output_dir'] = str(output_interface_dir)
        output_interface_dir.mkdir(parents=True, exist_ok=True)

        if "mlflow" in config['interfaces']:
            mlflow_dir = Path(config['utils']['input_dir']) / config['interfaces']['mlflow']['mlflow_dir']
            config['interfaces']['mlflow']['mlflow_dir'] = str(mlflow_dir.resolve())
            mlflow_dir.mkdir(parents=True, exist_ok=True)

        # Init service dir
        output_service_dir = output_dir / config['output_dir']
        config['output_dir'] = str(output_service_dir)
        output_service_dir.mkdir(parents=True, exist_ok=True)

        # Init logger
        Logger(config=config['utils']['logger'], filename="main")
        logger = logging.getLogger(__name__)
        logger.info("Start main")

        if config_file is not None:
            # Copy the config file into the output folder
            copy(config_file_path, config_dir)
            # Save also utils config file with full path for any inconvenience
            filename = Path(config_file_path).stem + "_fullpaths.json"
            file_path = os.path.join(config_dir, filename)
            with open(file_path, 'w') as json_file:
                json.dump(config, json_file, indent=4)

        elif config_json is not None:
            # Write config into the output folder
            file_path = os.path.join(config_dir, "api_conf_file.json")
            with open(file_path, 'w') as json_file:
                json.dump(config, json_file, indent=4)

        # Init service
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

    # Set configuration
    args = arg_parser.parse_args()

    output = main(config_file=args.c)



