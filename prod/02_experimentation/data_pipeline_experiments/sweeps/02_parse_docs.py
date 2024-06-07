# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse documents and save to Silver table
# MAGIC
# MAGIC This notebook will load new raw documents from the Bronze table and parses them using the parsing method specified in the configuration notebook.

# COMMAND ----------

# DBTITLE 1,THIS CELL NEEDS TO RUN FIRST
# MAGIC %run ./parser_library

# COMMAND ----------

import warnings
import mlflow 
import pyspark.sql.functions as func

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

destination_tables_config.get("raw_files_table_name")

# COMMAND ----------

# Get MLflow run
run = get_or_start_mlflow_run(MLFLOW_EXPERIMENT_NAME, mlflow_run_name)

# COMMAND ----------

# DBTITLE 1,Load Bronze Table as Stream
df_raw_bronze = spark.readStream.table(
    destination_tables_config.get("raw_files_table_name")
).select("path", "content")

# COMMAND ----------

# MAGIC %md
# MAGIC Once the Bronze table with the new raw PDF documents has been loaded, we can apply the configured parser. The subsequent cell utilizes the `parser-factory` to obtain a UDF of the parser function and applies the UDF to the `content` column, which contains the PDF binaries.

# COMMAND ----------

# DBTITLE 1,Apply parsing UDF
parser_udf = parser_factory(pipeline_config)
df_parsed_silver = (
    df_raw_bronze.withColumn("parsing", parser_udf("content"))
).drop("content")

# COMMAND ----------

# MAGIC %md
# MAGIC The parser functions, implemented in the `parser_library` notebook, handle exceptions in cases where a parser might fail for a specific document. Successful cases are marked with `SUCCESS` in the `parser_status` field of the `parsing` column, while failed attempts contain the error message.
# MAGIC
# MAGIC To filter out the failed cases, we extract the successful ones and append them to a Silver table in the following cell.

# COMMAND ----------

# DBTITLE 1,Append new documents to silver table
# Append successfully parsed documents to silver
silver_success_query = (
    df_parsed_silver.filter(df_parsed_silver.parsing.parser_status == "SUCCESS")
    .withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents"))
    .drop("parsing")
    .writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('parsed_docs_table_name').split('.')[-1]}",
    )
    .toTable(destination_tables_config.get("parsed_docs_table_name"))
)

silver_success_query.awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC We don't want to silently discard any potentially unsuccessful cases where documents could not be parsed.
# MAGIC
# MAGIC To [monitor the streaming query](https://www.databricks.com/blog/2022/05/27/how-to-monitor-streaming-queries-in-pyspark.html) for failed parsing attempts, we incorporate a custom metric using [`df.observe()`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.observe.html). This metric counts all rows where `parser_status != "SUCCESS"` and raises a warning if the count exceeds 0.
# MAGIC
# MAGIC Additionally, we append the respective rows to a `quarantine` table.

# COMMAND ----------

# DBTITLE 1,Save unsuccessful attempts to quarantine
df_parsed_quarantine = (
    df_parsed_silver.filter(df_parsed_silver.parsing.parser_status != "SUCCESS")
    .withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents"))
    .drop("parsing")
    .observe("parser_metric", func.count(func.lit(1)).alias("cnt_parsing_failures"))
)

silver_failures_query = (
    df_parsed_quarantine.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('parsed_docs_table_name').split('.')[-1]}_quarantine",
    )
    .toTable(f'{destination_tables_config.get("parsed_docs_table_name")}_quarantine')
)

silver_failures_query.awaitTermination()

# Warn if any parsing attempt failed
try:
    if silver_failures_query.lastProgress["observedMetrics"]["parser_metric"][
        "cnt_parsing_failures"
    ]:
        warnings.warn(
            f"The parsing failed for {silver_failures_query.lastProgress.get('numInputRows')} documents"
        )
except KeyError as e:
    print(f"Failed to identify if any parsing records failed: {e}")

# COMMAND ----------

tag_delta_table(destination_tables_config.get("parsed_docs_table_name"), configuration)

mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("parsed_docs_table_name")), context="parsed_docs")

# COMMAND ----------

dbutils.notebook.exit(True)
