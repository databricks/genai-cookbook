# Databricks notebook source
# MAGIC %md
# MAGIC ## Shared configuration
# MAGIC
# MAGIC This notebook initializes a `AgentCookbookConfig` Pydantic class to define parameters that are shared between the cookbook notebooks:
# MAGIC - Unity Catalog schema 
# MAGIC - MLflow Experiment to track versions of the Agent and their associated quality/cost/latency evaluation results
# MAGIC - Unity Catalog model that stores versions of the Agent's code/config that are deployed
# MAGIC - Evaluation Set Delta Table
# MAGIC
# MAGIC This notebook does the following:
# MAGIC 1. Creates the UC catalog/schema if they don't exist
# MAGIC 2. Serializes the configuration to `config/cookbook_config.yaml` so other notebooks can load it
# MAGIC
# MAGIC **Important: We suggest starting from a fresh clone of the cookbook if you need to change this configuration.**

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
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Get user info to set default values

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Get current user's name & email 
w = WorkspaceClient()
user_email = w.current_user.me().user_name
user_name = user_email.split("@")[0].replace(".", "_")

# Get the workspace default UC catalog
default_catalog = spark.sql("select current_catalog() as cur_catalog").collect()[0]['cur_catalog']

# COMMAND ----------

# MAGIC %md
# MAGIC ### ✅✏️ Configure this instance of the cookbook

# COMMAND ----------

from cookbook_utils.cookbook_config import AgentCookbookConfig

shared_config = AgentCookbookConfig(
    uc_catalog_name=f"{default_catalog}",
    uc_schema_name=f"{user_name}_agents", 
    uc_asset_prefix="agent_app_name", # Prefix to every created UC asset, typically the Agent's name
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 🚫✏️ Save the configuration for use by other notebooks

# COMMAND ----------

# Save configuration
shared_config.dump_to_yaml('./configs/cookbook_config.yaml')

# Print configuration
shared_config.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Validate the storage locations exist, create if they don't exist

# COMMAND ----------

if not shared_config.validate_or_create_uc_catalog():
  raise Exception("UC Catalog is not valid, fix per the console notes above.")

if not shared_config.validate_or_create_uc_schema():
  raise Exception("UC Schema is not valid, fix per the console notes above.")

if not shared_config.validate_or_create_mlflow_experiment():
  raise Exception("MLflow experiment name is not valid, fix per the console notes above.")
