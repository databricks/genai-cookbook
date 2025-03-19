import logging
from typing import List
from cookbook.config import SerializableConfig
import yaml
import mlflow
from mlflow.models import ModelConfig
from cookbook.config import (
    load_serializable_config_from_yaml,
)
import os


def load_first_yaml_file(config_paths: List[str]) -> str:
    for path in config_paths:
        if os.path.exists(path):
            logging.info(f"Found YAML config file at {path}")
            with open(path, "r") as handle:
                return handle.read()
    raise ValueError(
        f"No config file found at any of the following paths: {config_paths}. "
        f"Please ensure a config file exists at one of those paths."
    )


def load_config_from_mlflow_model_config() -> SerializableConfig:
    try:
        model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
        loaded_config = load_serializable_config_from_yaml(model_config_as_yaml)
        logging.info(f"Loaded config from mlflow.models.ModelConfig(): {loaded_config}")
        return loaded_config
    except Exception as e:
        logging.info(f"Could not load config from mlflow.models.ModelConfig(): {e}")
        return None


def try_to_load_config_file(agent_config_file_or_path: str) -> SerializableConfig:
    """
    Try to load configuration from a local YAML file.
    """

    # otherwise, we try to look for the YAML file
    # this logic accounts for the fact that the agent can be called from any working directory, so we have to search for the config folder to find the YAML.
    config_paths = []
    config_paths.append(
        agent_config_file_or_path
    )  # will try from the passed location first.

    # Then try a from a few common locations - these are set based on the common working directory locations for a notebook/shell.
    config_paths.extend(
        [
            "./configs/" + agent_config_file_or_path,
            "../configs/" + agent_config_file_or_path,
            "../../configs/" + agent_config_file_or_path,
            "../openai_sdk_agent_app_sample_code/configs/" + agent_config_file_or_path,
            "./openai_sdk_agent_app_sample_code/configs/" + agent_config_file_or_path,
        ]
    )

    logging.info(
        f"Trying to load YAML file {agent_config_file_or_path} from paths: {config_paths}"
    )
    try:
        config_file = load_first_yaml_file(config_paths)
        return load_serializable_config_from_yaml(config_file)
    except Exception as e:
        logging.info(
            f"Exception loading YAML file {agent_config_file_or_path} at {config_paths}: {e}"
        )
        raise ValueError(
            f"Could not load the provided YAML file {agent_config_file_or_path}."
        )


def load_config(
    passed_agent_config: SerializableConfig | str | None = None,
    default_config_file_name: str = None,
) -> SerializableConfig:
    """
    Load configuration from various sources in order of precedence:
    # load the Agent's configuration.  Priority order:
    1. MLflow Model config
    2. passed_agent_config
    3. default_config_file_name

    Returns:
        SerializableModel: Loaded configuration object
    """

    # 1. Try to use MLflow ModelConfig
    try:
        logging.info("Trying to load config from mlflow.models.ModelConfig()")
        model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
        loaded_config = load_serializable_config_from_yaml(model_config_as_yaml)
        logging.info(f"Loaded config from mlflow.models.ModelConfig(): {loaded_config}")
        return loaded_config
    except FileNotFoundError as e:
        logging.info(f"Could not load config from mlflow.models.ModelConfig(): {e}")

    # 2a. passed_agent_config is an instantiated config class, use that
    if isinstance(passed_agent_config, ModelConfig):
        logging.info(
            "passed_agent_config` is an instantiated config class, using that."
        )
        return passed_agent_config

    # 2b. passed_agent_config is a YAML file name or file path, try to load from that YAML file
    # try_to_load_config_file logic accounts for the fact that the agent can be called from any working directory, so we will search for the config folder to find the YAML.
    if isinstance(passed_agent_config, str):
        print("ENTRO AQUI")
        logging.info(
            f"`passed_agent_config` is a string, trying to load from YAML: {passed_agent_config}"
        )
        try:
            loaded_config = try_to_load_config_file(passed_agent_config)
            logging.info(
                f"Loaded config from YAML file {passed_agent_config}: {loaded_config}"
            )
            return loaded_config
        except ValueError as e:
            logging.info(f"{passed_agent_config} was not found.")

    # 3. Try to load from default config file
    if default_config_file_name:
        logging.info(f"Trying to load from YAML: {default_config_file_name}")
        try:
            loaded_config = try_to_load_config_file(default_config_file_name)
            logging.info(
                f"Loaded config from YAML file {default_config_file_name}: {loaded_config}"
            )
            return loaded_config
        except ValueError as e:
            logging.info(f"{default_config_file_name} was not found.")

    # If no config is found so far, return None
    logging.error(
        "load_config could not find a config file.  Returning None.  Refer to your Agent's error message for next steps."
    )
    return None
