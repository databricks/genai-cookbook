# Databricks notebook source
# MAGIC %md
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC ### Step 1: Agent storage configuration
# MAGIC
# MAGIC This notebook initializes a `AgentStorageConfig` Pydantic class to define the locations where the Agent's code/config and its supporting data & metadata is stored in the Unity Catalog:
# MAGIC - **Unity Catalog Model:** Stores staging/production versions of the Agent's code/config
# MAGIC - **MLflow Experiment:** Stores every development version of the Agent's code/config, each version's associated quality/cost/latency evaluation results, and any MLflow Traces from your development & evaluation processes
# MAGIC - **Evaluation Set Delta Table:** Stores the Agent's evaluation set
# MAGIC
# MAGIC This notebook does the following:
# MAGIC 1. Validates the provided locations exist.
# MAGIC 2. Serializes this configuration to `config/agent_storage_config.yaml` so other notebooks can use it

# COMMAND ----------

# MAGIC %md
# MAGIC **Important note:** Throughout this notebook, we indicate which cells you:
# MAGIC - ✅✏️ *should* customize - these cells contain config settings to change
# MAGIC - 🚫✏️ *typically will not* customize - these cells contain boilerplate code required to validate / save the configuration
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Install Python libraries

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Connect to Databricks
# MAGIC
# MAGIC If running locally in an IDE using Databricks Connect, connect the Spark client & configure MLflow to use Databricks Managed MLflow.  If this running in a Databricks Notebook, these values are already set.

# COMMAND ----------

from mlflow.utils import databricks_utils as du
import os
if not du.is_in_databricks_notebook():
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Get current user info to set default values

# COMMAND ----------

from cookbook.databricks_utils import get_current_user_info

user_email, user_name, default_catalog = get_current_user_info(spark)

print(f"User email: {user_email}")
print(f"User name: {user_name}")
print(f"Default UC catalog: {default_catalog}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅✏️ Configure your Agent's storage locations
# MAGIC
# MAGIC Either review & accept the default values or enter your preferred location.

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.databricks_utils import get_mlflow_experiment_url
import mlflow

# Default values below for `AgentStorageConfig` 
agent_name = "my_agent_autogen"
uc_catalog_name = "casaman_ssa"
uc_schema_name = "demos"

# Agent storage configuration
agent_storage_config = AgentStorageConfig(
    uc_model_name=f"{uc_catalog_name}.{uc_schema_name}.{agent_name}",  # UC model to store staging/production versions of the Agent's code/config
    evaluation_set_uc_table=f"{uc_catalog_name}.{uc_schema_name}.{agent_name}_eval_set",  # UC table to store the evaluation set
    mlflow_experiment_name=f"/Users/{user_email}/{agent_name}_mlflow_experiment",  # MLflow Experiment to store development versions of the Agent and their associated quality/cost/latency evaluation results + MLflow Traces
)

# Validate the UC catalog and schema for the Agent'smodel & evaluation table
is_valid, msg = agent_storage_config.validate_catalog_and_schema()
if not is_valid:
    raise Exception(msg)

# Set the MLflow experiment, validating the path is valid
experiment_info = mlflow.set_experiment(agent_storage_config.mlflow_experiment_name)
# If running in a local IDE, set the MLflow experiment name as an environment variable
os.environ["MLFLOW_EXPERIMENT_NAME"] = agent_storage_config.mlflow_experiment_name

print(f"View the MLflow Experiment `{agent_storage_config.mlflow_experiment_name}` at {get_mlflow_experiment_url(experiment_info.experiment_id)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Save the configuration for use by other notebooks

# COMMAND ----------

from cookbook.config import serializable_config_to_yaml_file

serializable_config_to_yaml_file(agent_storage_config, "./configs/agent_storage_config.yaml")