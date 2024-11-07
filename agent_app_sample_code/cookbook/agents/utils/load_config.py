import logging
from typing import List
from cookbook.config import SerializableConfig
import yaml
import mlflow
from cookbook.config import (
    load_serializable_config_from_yaml_file,
)
from cookbook.config import (
    load_serializable_config_from_yaml,
)
import os


def find_project_root(marker_directory="configs"):
    """Find the project root by looking for the "configs" directory."""
    current = os.path.abspath(os.getcwd())
    while current != "/":
        # Check current directory
        marker_path = os.path.join(current, marker_directory)
        if os.path.exists(marker_path) and os.path.isdir(marker_path):
            return current

        # Check immediate subdirectories
        for item in os.listdir(current):
            item_path = os.path.join(current, item)
            if os.path.isdir(item_path):
                marker_in_subdir = os.path.join(item_path, marker_directory)
                if os.path.exists(marker_in_subdir) and os.path.isdir(marker_in_subdir):
                    return current

                # Check one level deeper
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        marker_in_subsubdir = os.path.join(
                            subitem_path, marker_directory
                        )
                        if os.path.exists(marker_in_subsubdir) and os.path.isdir(
                            marker_in_subsubdir
                        ):
                            return current

        current = os.path.dirname(current)
    raise ValueError(f"Could not find project root containing {marker_directory}")


def load_first_yaml_file(config_paths: List[str]) -> str:
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as handle:
                return handle.read()
    raise ValueError(
        f"No config file found at any of the following paths: {config_paths}. "
        f"Please ensure a config file exists at one of those paths."
    )


def load_config(
    agent_config: SerializableConfig | str | None = None,
    default_config_file_name: str = None,
) -> SerializableConfig:
    """
    Load configuration from various sources in order of precedence:
    1. Passed config object
    2. YAML file path
    3. Temporary config file
    4. Default config files
    5. MLflow model config

    Args:
        agent_config: Configuration object, path to YAML file, or None

    Returns:
        SerializableModel: Loaded configuration object
    """
    if isinstance(agent_config, str):
        logging.info(
            f"`agent_config` is a string, trying to load from YAML: {agent_config}"
        )
        return load_serializable_config_from_yaml_file(agent_config)

    if isinstance(agent_config, SerializableConfig):
        logging.info("Passed instantiated config class, using that.")
        return agent_config

    # Try to load from default config file first for inner dev loop
    # in serving env these files will not be present, so load the model's logged config e.g., the config from mlflow.pyfunc.log_model(model_config=...) via mlflow.ModelConfig()
    # in the shared logging utilities, we set TMP_CONFIG_FILE_PATH to the path of the config file that is dumped
    config_paths = []

    if default_config_file_name:
        try:
            project_root = find_project_root()
            config_paths.extend(
                [
                    os.path.join(project_root, "configs", default_config_file_name),
                    os.path.join(
                        project_root,
                        "agent_app_sample_code",
                        "configs",
                        default_config_file_name,
                    ),
                ]
            )
        except ValueError as e:
            # could not find the project root, so keep trying the next options to load config
            logging.info(
                f"Could not find project root directory by looking for `configs` directory.  Trying to load config from current working directory {os.getcwd()}."
            )
            config_paths.extend(
                [
                    "./configs/" + default_config_file_name,
                    "../configs/" + default_config_file_name,
                    "../../configs/" + default_config_file_name,
                    "../agent_app_sample_code/configs/" + default_config_file_name,
                    "./agent_app_sample_code/configs/" + default_config_file_name,
                ]
            )
            pass
    logging.info(f"Trying to load from paths: {config_paths}")
    try:
        config_file = load_first_yaml_file(config_paths)
        return load_serializable_config_from_yaml(config_file)
    except ValueError as e:
        logging.info(
            f"No local config YAML found at {config_paths}, trying mlflow.ModelConfig()"
        )
        # TODO: replace with mlflow.ModelConfig().to_dict() once released
        # model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
        try:
            model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
            config = load_serializable_config_from_yaml(model_config_as_yaml)
            logging.info(f"Loaded config from mlflow.ModelConfig(): {config}")
            return config
        except Exception as e:
            logging.error(f"Error loading config from mlflow.ModelConfig(): {e}")
            return None

    # If no config is found, return None
    return None
