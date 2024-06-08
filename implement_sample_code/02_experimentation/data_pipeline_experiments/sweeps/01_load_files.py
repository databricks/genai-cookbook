# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Documents as Binaries to a Bronze Table
# MAGIC New PDF files are loaded as one binary per row and appended to the bronze table. Additional metadata, such as `modificationTime` and `length` (i.e., file size), accompanies each entry.

# COMMAND ----------

# MAGIC %run ./shared_utilities

# COMMAND ----------

# MAGIC %md
# MAGIC By default, get the strategy from 00_config in Try Single Strategy mode.
# MAGIC This will be overwritten in the next cell if running in Sweep Strategies mode.

# COMMAND ----------

# DBTITLE 1,Load Configuration Dict
# MAGIC %run ./00_config

# COMMAND ----------

# Allow for override of `00_config` during the sweep
dbutils.widgets.text("strategy_to_run", "", "JSON string of strategy")
vectorsearch_config, embedding_config,pipeline_config, destination_tables_config, configuration, mlflow_run_name = load_strategy_from_widget(dbutils.widgets.get("strategy_to_run"))

# COMMAND ----------

import mlflow 
# Start MLflow logging
run = get_or_start_mlflow_run(MLFLOW_EXPERIMENT_NAME, mlflow_run_name)

# Tag the run
mlflow.set_tag("type", "data_pipeline")

# Set the parameters
mlflow.log_params(_flatten_nested_params({"data_pipeline": configuration}))
mlflow.log_params(_flatten_nested_params({"destination_tables": destination_tables_config}))

# Log the configs as artifacts for later use
mlflow.log_dict(destination_tables_config, "destination_tables_config.json")
mlflow.log_dict(configuration, "data_pipeline_config.json")

# COMMAND ----------

# DBTITLE 1,Load new pdf files from source volume. We are using the [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) which has two [file discovery modes](https://docs.databricks.com/en/ingestion/auto-loader/file-detection-modes.html) for ingestion. Here, we are using the [Directory Listing](https://docs.databricks.com/en/ingestion/auto-loader/file-detection-modes.html#directory-listing-mode). For high file volume scenarios the [File Notifications](https://docs.databricks.com/en/ingestion/auto-loader/file-detection-modes.html#file-notification-mode) mode will be more performant.
df_raw_bronze = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobfilter", f"*.{pipeline_config.get('file_format')}")
    .load(SOURCE_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC The subsequent cell appends rows with new PDF binaries to the Bronze table. It's essential to note that we utilize [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) to incrementally load and process new files. Thus, we must specify a checkpoint location to keep track of the processed input files.

# COMMAND ----------

# DBTITLE 1,Append new pdf binaries to Bronze Table
df_raw_bronze.writeStream.trigger(availableNow=True).option(
    "checkpointLocation",
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('raw_files_table_name').split('.')[-1]}",
).toTable(destination_tables_config.get("raw_files_table_name")).awaitTermination()

# COMMAND ----------

tag_delta_table(destination_tables_config.get("raw_files_table_name"), configuration)

mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("raw_files_table_name")), context="raw_files")

# COMMAND ----------

dbutils.notebook.exit(True)
