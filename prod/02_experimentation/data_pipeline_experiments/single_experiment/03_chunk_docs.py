# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Documents and Save to Gold Table
# MAGIC
# MAGIC This notebook loads new parsed documents from the Silver table and chunks them using the chunking method specified in the configuration notebook. The results are then appended to a Gold table, forming the basis for the vector search index.

# COMMAND ----------

# DBTITLE 1,THIS CELL NEEDS TO RUN FIRST
# MAGIC %run ./chunker_library

# COMMAND ----------

import warnings
import pyspark.sql.functions as func
import mlflow

# COMMAND ----------

# DBTITLE 1,Load Configuration Dict
# MAGIC %run ./00_config

# COMMAND ----------

# Get MLflow run
run = get_or_start_mlflow_run(MLFLOW_EXPERIMENT_NAME, mlflow_run_name)

# COMMAND ----------

# DBTITLE 1,Load Silver table as Stream
df_parsed_silver = spark.readStream.table(
    destination_tables_config.get("parsed_docs_table_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC After loading the Silver table with the new parsed documents, we apply the configured chunker to split the document text into smaller chunks, preparing them for loading into a vector database.
# MAGIC
# MAGIC The subsequent cell utilizes the `chunker-factory` to obtain a UDF of the chunker function and applies the UDF to the `parsed_content` field of the `doc_parsed_contents` column, which contains the parsed text.
# MAGIC
# MAGIC The chunker UDF returns an array with all the chunks for each document. Before appending the chunks to the Gold table, we explode the array so that each chunk is associated with a single row. Additionally, because a vector search index requires a primary key, a column with the MD5 hash of the document chunk is added.

# COMMAND ----------

# DBTITLE 1,Apply chunking UDF
# Get configured chunker from chunker factory (see chunking_library notebook)
chunking_udf = chunker_factory(pipeline_config, embedding_config)

df_chunked = df_parsed_silver.withColumn(
    "chunked", chunking_udf("doc_parsed_contents")
)

# In order to later sync this table to the vector search index we need a primary key.
# For this demo we simply compute a md5 hash. Consider to define a proper primary key for
# production.
df_chunked_gold = df_chunked.filter(df_chunked.chunked.chunker_status == "SUCCESS").select(
    "path",
    func.explode("chunked.chunked_text").alias("chunked_text"),
    func.md5(func.col("chunked_text")).alias("chunk_id")
)


if pipeline_config['chunker']['output_table']['include_parser_metadata_as_columns']:

    keys_df = spark.read.table(
    destination_tables_config.get("parsed_docs_table_name")
    )
    
    df_keys_array = keys_df.select(func.explode(func.map_keys(keys_df.doc_parsed_contents)).alias("key")).distinct().toPandas()["key"].tolist()

    metadata_df = df_parsed_silver

    if pipeline_config['chunker']['output_table']['include_parent_doc_content_as_column'] == False:
        if 'parsed_content' in df_keys_array:
            df_keys_array.remove('parsed_content')

    for key in df_keys_array:
        metadata_df = metadata_df.withColumn(key, metadata_df.doc_parsed_contents[key])

    metadata_df = metadata_df.drop("doc_parsed_contents").withColumnRenamed("path", "path_2")

    df_chunked_gold = df_chunked_gold.join(metadata_df, func.expr("path_2 = path"), "inner").drop("path_2")

# COMMAND ----------

# MAGIC %md
# MAGIC The chunker functions, implemented in the `chunker_library` notebook, handle exceptions in cases where a chunker might fail for a specific document. Successful cases are marked with `SUCCESS` in the `chunker_status` field of the `chunked` column, while failed attempts contain the error message.
# MAGIC
# MAGIC To filter out the failed rows, we apply a filter to extract the successful ones and append them to the Gold table in the following cell.

# COMMAND ----------

# DBTITLE 1,Append new, parsed and chunked documents to Silver
chunking_query = (
    df_chunked_gold.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('chunked_docs_table_name').split('.')[-1]}",
    )
    .toTable(destination_tables_config.get("chunked_docs_table_name"))
)
chunking_query.awaitTermination()

# change data feed is required for sync to vector search
spark.sql(f"ALTER TABLE {destination_tables_config.get('chunked_docs_table_name')} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC We don't want to silently discard any potentially unsuccessful cases where documents could not be chunked.
# MAGIC
# MAGIC To [monitor the streaming query](https://www.databricks.com/blog/2022/05/27/how-to-monitor-streaming-queries-in-pyspark.html) for failed chunking attempts, we incorporate a custom metric using [`df.observe()`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.observe.html). This metric counts all rows where `chunker_status != "SUCCESS"` and raises a warning if the count exceeds 0.
# MAGIC
# MAGIC Additionally, we append the respective rows to a `quarantine` table.

# COMMAND ----------

# DBTITLE 1,Save unsuccessful attempts to quarantine
df_chunked_quarantine = df_chunked.filter(df_chunked.chunked.chunker_status != "SUCCESS").observe(
    "chunking_metric", func.count(func.lit(1)).alias("cnt_chunking_failures"))

chunking_quarantine_query = (
    df_chunked_quarantine.writeStream.trigger(availableNow=True)
    .option(
        "checkpointLocation",
        f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('chunked_docs_table_name').split('.')[-1]}_quarantine",
    )
    .toTable(f'{destination_tables_config.get("chunked_docs_table_name")}_quarantine')
)
chunking_quarantine_query.awaitTermination()

# Warn if any parsing attempt failed
try:
    if chunking_quarantine_query.lastProgress["observedMetrics"]["chunking_metric"]["cnt_chunking_failures"]>0:
        warnings.warn(
            f'The parsing failed in {chunking_quarantine_query.lastProgress["observedMetrics"]["chunking_metric"]["cnt_chunking_failures"]} cases'
        )
except KeyError as e:
    print(f"Failed to identify if any chunking records failed: {e}")

# COMMAND ----------

spark.read.table(destination_tables_config.get('chunked_docs_table_name')).display()

# COMMAND ----------

tag_delta_table(destination_tables_config.get("chunked_docs_table_name"), configuration)

mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("chunked_docs_table_name")), context="chunked_docs")
