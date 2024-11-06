import logging
from typing import Dict
import yaml
import mlflow
from utils.agents.tools import (
    SerializableModel,
    load_obj_from_yaml_file,
    load_obj_from_yaml,
)
from utils.agents.yaml_loader import load_first_yaml_file

TMP_CONFIG_FILE_NAME = "TMP_CONFIG_FILE_NAME"


def load_config(
    agent_config: SerializableModel | str | None = None,
    default_config_file_name: str = None,
) -> SerializableModel:
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
        return load_obj_from_yaml_file(agent_config)

    if isinstance(agent_config, SerializableModel):
        logging.info("Passed instantiated config class, using that.")
        return agent_config

    # Try to load from default config file first for inner dev loop
    # in serving env these files will not be present, so load the model's logged config e.g., the config from mlflow.pyfunc.log_model(model_config=...) via mlflow.ModelConfig()
    # in the shared logging utilities, we set TMP_CONFIG_FILE_PATH to the path of the config file that is dumped
    config_paths = []
    # tmp_config_file_path = globals().get(TMP_CONFIG_FILE_NAME, None)
    # print(f"tmp_config_file_path: {tmp_config_file_path}")

    # if tmp_config_file_path:
    #     config_paths.append(tmp_config_file_path)

    if default_config_file_name:
        config_paths.extend(
            [
                f"../../configs/{default_config_file_name}",
                f"./configs/{default_config_file_name}",
            ]
        )

    try:
        config_file = load_first_yaml_file(config_paths)
        return load_obj_from_yaml(config_file)
    except ValueError as e:
        logging.info(
            f"No local config YAML found at {config_paths}, trying mlflow.ModelConfig()"
        )
        # TODO: replace with mlflow.ModelConfig().to_dict() once released
        # model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
        model_config_as_yaml = yaml.dump(
            mlflow.models.ModelConfig(
                development_config={"test": "test"}
            )._read_config()
        )
        config = load_obj_from_yaml(model_config_as_yaml)
        logging.info(f"Loaded config from mlflow.ModelConfig(): {config}")
        return config

    # If no config is found, return None
    return None


@mlflow.trace()
def remove_message_keys_with_null_values(message: Dict[str, str]) -> Dict[str, str]:
    """
    Remove any keys with None/null values from the message.
    Having a null value for a key breaks DBX model serving input validation even if that key is marked as optional in the schema, so we remove them.
    Example: refusal key is set as None by OpenAI
    """
    return {k: v for k, v in message.items() if v is not None}
