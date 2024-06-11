# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny

# COMMAND ----------

# MAGIC %run ./shared_utilities

# COMMAND ----------

# MAGIC %run ../../../00_global_config

# COMMAND ----------

# MAGIC %md ## Configuration for this pipeline
# MAGIC
# MAGIC 1. `CHECKPOINTS_VOLUME_PATH`: Unity Catalog Volume to store Spark streaming checkpoints

# COMMAND ----------

# Temporary location to store Spark streaming checkpoints
# This must a UC Volume; suggest keeping as the default value.
CHECKPOINTS_VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/checkpoints"
