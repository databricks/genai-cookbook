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

# MAGIC %pip install -qqqq -U 'pydantic>=2.9.2' mlflow databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/agent_storage_location_config

# COMMAND ----------

# Get current user's name for the default UC catalog name
user_name = spark.sql("SELECT current_user() as username").collect()[0].username.split("@")[0].replace(".", "").lower()[:35]

# COMMAND ----------

# Agent's storage location configuration
agent_storage_locations_config = AgentStorageLocationConfig(
    uc_catalog=f"{user_name}_catalog", 
    uc_schema="agents", 
    uc_asset_prefix="agent_app_name", # Prefix to every created UC asset
)

# Save configuration
agent_storage_locations_config.dump_to_yaml('./configs/agent_config.yaml')

# Print configuration
agent_storage_locations_config.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate the configuration's locations

# COMMAND ----------

# MAGIC %run ./validators/validate_agent_storage_locations
