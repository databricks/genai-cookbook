# Databricks notebook source
# MAGIC %md
# MAGIC #How to use
# MAGIC
# MAGIC 1. Configure your data processing strategy in `00_config`
# MAGIC 2. Walk through the instructions in this Notebook to configure your data pipeline
# MAGIC 3. If needed, resolve any identified configuration errors and re-run

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate configuration

# COMMAND ----------

# MAGIC %run ./00_validate_config

# COMMAND ----------

# MAGIC %md ## Delete existing tables & indexes for the configuration
# MAGIC
# MAGIC If you have changed the code or configuration for an configuration that you already ran, you need to run this cell or Spark will not re-process already processed files.

# COMMAND ----------

dbutils.notebook.run("reset_tables_and_checkpoints", timeout_seconds=0)

# COMMAND ----------

# MAGIC %md ## Run the entire pipeline

# COMMAND ----------

dbutils.notebook.run("01_load_files", timeout_seconds=0)

# COMMAND ----------

dbutils.notebook.run("02_parse_docs", timeout_seconds=0)

# COMMAND ----------

dbutils.notebook.run("03_chunk_docs", timeout_seconds=0)

# COMMAND ----------

dbutils.notebook.run("04_vector_index", timeout_seconds=0)

# COMMAND ----------

print(f"MLflow Run name to use in the evaluation notebook: {mlflow_run_name}")
