# Databricks notebook source
# MAGIC %md
# MAGIC ## Global configuration
# MAGIC
# MAGIC This notebook initializes a `AgentStorageLocationConfig` Pydantic class to define the storage locations that are shared between the cookbook notebooks. Storage locations include: 
# MAGIC - Unity Catalog schema to store the Agent's Delta Table, Vector Index, and Model resources.
# MAGIC - MLflow Experiment to track versions of the Agent and their associated quality/cost/latency evaluation results
# MAGIC
# MAGIC This notebook does the following:
# MAGIC 1. Creates the UC catalog/schema if they don't exist
# MAGIC 2. Serializes the configuration to `config/agent_storage_locations.yaml` so other notebooks can load it
# MAGIC
# MAGIC
# MAGIC **Important: If you need to change this configuration after running the included notebooks, make sure re-run the notebooks because later notebooks depend upon data created in earlier notebooks.**
# MAGIC

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Get current user's name for the default UC schema
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "_")

# Get the workspace default UC catalog
default_catalog = spark.sql("select current_catalog() as cur_catalog").collect()[0]['cur_catalog']

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modify this cell to set your Agent's storage locations

# COMMAND ----------

from utils import AgentStorageLocationConfig

# Agent's storage location configuration
agent_storage_locations_config = AgentStorageLocationConfig(
    uc_catalog=f"{default_catalog}", 
    uc_schema=f"{user_name}_agents", 
    uc_asset_prefix="agent_app_name", # Prefix to every created UC asset, typically the Agent's name
)

# COMMAND ----------

# Save configuration
agent_storage_locations_config.dump_to_yaml('./configs/agent_config.yaml')

# Print configuration
agent_storage_locations_config.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate the configuration's locations

# COMMAND ----------

from utils import validate_storage_config

validate_storage_config(agent_storage_locations_config)
