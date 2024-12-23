import logging
import yaml
from pathlib import Path
import os 
import os.path as path
from box import Box

# Load the function calling Agent's configuration
fc_agent_config = Box(yaml.safe_load(Path("./configs/function_calling_agent_config.yaml").read_text()))
print(fc_agent_config)

def load_config():
    try:
        fc_agent_config_path =  path.abspath(path.join(__file__ ,"../../../../configs/function_calling_agent_config.yaml"))
        logging.info(f"Trying to load config from {fc_agent_config_path}")
        print(fc_agent_config_path)
        agent_conf = Box(yaml.safe_load(Path(fc_agent_config_path).read_text()))
        return agent_conf
    except FileNotFoundError as e:
        return load_config_from_mlflow_model_config()

def load_config_from_mlflow_model_config():
    try:
        logging.info("Trying to load config from mlflow.models.ModelConfig()")
        model_config_as_yaml = yaml.dump(mlflow.models.ModelConfig()._read_config())
        loaded_config = load_serializable_config_from_yaml(model_config_as_yaml)
        logging.info(f"Loaded config from mlflow.models.ModelConfig(): {loaded_config}")
        return loaded_config
    except FileNotFoundError as e:
        logging.info(f"Could not load config from mlflow.models.ModelConfig(): {e}")