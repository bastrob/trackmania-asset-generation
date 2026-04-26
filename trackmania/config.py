import yaml
from loguru import logger


def load_config(config_path: str):
    """
    Load a yaml configuration file
    Args:
        config_path: The configuration path
    Returns a dict
    """
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            logger.error(f"Error in configuration file: {exc}")