# Databricks notebook source
# Helper function to merge the configurations
def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# COMMAND ----------

# Helper function to determine winning chains
import operator

def count_wins(config_name, metrics_to_use):
    return sum(1 for d in metrics_to_use if d['winner'] == config_name)

# COMMAND ----------

# Helper functions to load baseline chain code / configs to files
from databricks.sdk import WorkspaceClient
import base64
from databricks.sdk.service.workspace import ExportFormat
import os
from databricks.sdk.service.workspace import ImportFormat, Language
import yaml

def write_baseline_chain_code_to_notebook(chain_code_as_text, workspace_client, save_folder="baseline_chain", save_file_name="chain", overwrite=True):
    # Encode the content in base64
    encoded_content = base64.b64encode(chain_code_as_text.encode()).decode()

    # Load to the workspace as a Notebook that can be run
    save_path = f'{os.getcwd()}/{save_folder}/'
    # Check if the directory exists
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    output_location = f"{save_path}/{save_file_name}"
    workspace_client.workspace.import_(
        content=encoded_content,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=overwrite,
        path=output_location,
    )


def write_baseline_chain_config_to_yaml(chain_code_as_dict, save_folder="baseline_chain", save_file_name="rag_chain_config.yaml", overwrite=True):
    
    # Load to the workspace as a Notebook that can be run
    save_path = f'{os.getcwd()}/{save_folder}/'
    # Check if the directory exists
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    output_location = f"{save_path}/{save_file_name}"

    with open(output_location, 'w') as file:
        yaml.dump(chain_code_as_dict, file)
        

# COMMAND ----------

import mlflow

def get_mlflow_run(experiment_name, run_name):
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"run_name = '{run_name}'", output_format="list")

    if len(runs) != 1:
        raise ValueError(f"Found {len(runs)} runs with name {run_name}.  {run_name} must identify a single run.  Alternatively, you can adjust this code to search for a run based on `run_id`")

    return runs[0]

# COMMAND ----------

from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items
