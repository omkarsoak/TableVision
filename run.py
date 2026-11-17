import sys
import yaml
import logging
import argparse

# Set up logging configuration
logging.addLevelName(35, 'INFO')
logging.basicConfig(level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')


# Modules are imported lazily, so a module's heavy dependencies are only needed when that module runs.
def _inference(params):
    from src.inference import main
    return main(**params)


def _merge_and_upload(params):
    from src.merge_and_upload import main
    return main(**params)


def _granite_vision_finetune(params):
    from src.models.granite_vision.finetune import main
    return main(**params)


def run(config_path: str):
    """
    Run modules based on the configuration file.
    :param config_path: Path to the YAML configuration file.
    """
    # Load the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    modules = config.get('modules', [])
    if not modules:
        logging.error("No modules configuration found in the YAML file.")
        sys.exit(1)

    # Map module names to their functions
    module_map = {
        'inference': _inference,
        'merge_and_upload': _merge_and_upload,
        'granite_vision_finetune': _granite_vision_finetune,
    }

    try:
        for step in modules:
            module = step.get('module')
            run = step.get('run', False)
            params = step.get('params', {})

            if not run:
                logging.log(35, f"Skipping module: {module}...")
                continue

            if module not in module_map:
                logging.error(f"Unknown module: {module}")
                continue

            logging.log(35, f"Running module: {module}{f' with params: {params}' if params else ''}...")
            module_map[module](params)

        logging.log(35, "Modules execution completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the module execution: {e}", exc_info=True)
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the desired modules using a YAML configuration.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file (see configs/).")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args.config)
