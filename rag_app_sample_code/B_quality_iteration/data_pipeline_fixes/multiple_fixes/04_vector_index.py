# Databricks notebook source
# MAGIC %md
# MAGIC ## Synchronize Gold Table with Vector Search Index
# MAGIC
# MAGIC This notebook triggers the synchronization between the Gold table, which contains all document chunks, and the vector search index defined in the `vectorsearch_config`. If this is the first time, the index is created and deployed to the configured vector search endpoint.

# COMMAND ----------

# MAGIC %pip install -U databricks-vectorsearch pyyaml mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import datetime
import mlflow
from databricks.vector_search.client import VectorSearchClient

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

# Get MLflow run
run = get_or_start_mlflow_run(MLFLOW_EXPERIMENT_NAME, mlflow_run_name)

# COMMAND ----------

# DBTITLE 1,Get or Create VS endpoint
client = VectorSearchClient()
try:
  vs_endpoint = client.get_endpoint(vectorsearch_config.get("vectorsearch_endpoint_name"))
except Exception as e:
  print(f"Vectorsearch endpoint {vectorsearch_config.get('vectorsearch_endpoint_name')} is not available. Please choose a different endpoint.")

# COMMAND ----------

# DBTITLE 1,Sync or Create VS Index
try:
  index = client.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=destination_tables_config.get("vectorsearch_index_name"))
  print(f"Syncing index {destination_tables_config.get('vectorsearch_index_name')}")
  index.sync()
except Exception as e:
  print(f"The index {destination_tables_config.get('vectorsearch_index_name')} does not exist. Creating it now ...")
  index = client.create_delta_sync_index_and_wait(
      endpoint_name=VECTOR_SEARCH_ENDPOINT,
      index_name=destination_tables_config.get("vectorsearch_index_name"),
      primary_key="chunk_id",
      pipeline_type=vectorsearch_config.get("pipeline_type"),
      source_table_name=destination_tables_config.get('chunked_docs_table_name'),
      embedding_source_column="chunked_text",
      embedding_model_endpoint_name=embedding_config.get("embedding_endpoint_name"),
  )

# COMMAND ----------

tag_delta_table(destination_tables_config.get("vectorsearch_index_name"), configuration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output the data pipeline configuration for use in the chain

# COMMAND ----------

chain_config = {
    "databricks_resources": {
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
    },
    "retriever_config": {
        "vector_search_index": destination_tables_config[
            "vectorsearch_index_name"
        ],
        "data_pipeline_tag": configuration["strategy_short_name"],
    }
}

mlflow.log_dict(chain_config, "chain_config.json")

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

dbutils.notebook.exit(True)
