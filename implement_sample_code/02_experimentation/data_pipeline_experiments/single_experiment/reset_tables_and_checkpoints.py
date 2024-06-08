# Databricks notebook source
# MAGIC %pip install -U -qqq databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Delete tables created in 01_load_files
# delete table with raw documents
spark.sql(f'DROP TABLE IF EXISTS {destination_tables_config.get("raw_files_table_name")}')
# delete (and thus reset) corresponding checkpoint
dbutils.fs.rm(
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('raw_files_table_name').split('.')[-1]}",
    True,
)

# COMMAND ----------

# DBTITLE 1,Delete tables created in 02_parse_docs
# delete table with parsed documents
spark.sql(f'DROP TABLE IF EXISTS {destination_tables_config.get("parsed_docs_table_name")}')
# delete (and thus reset) corresponding checkpoint
dbutils.fs.rm(
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('parsed_docs_table_name').split('.')[-1]}",
    True,
)

# delete table with documents failed during parsing
spark.sql(
    f'DROP TABLE IF EXISTS {destination_tables_config.get("parsed_docs_table_name")}_quarantine'
)
# delete (and thus reset) corresponding checkpoint
dbutils.fs.rm(
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('parsed_docs_table_name').split('.')[-1]}_quarantine",
    True,
)

# COMMAND ----------

# DBTITLE 1,Delete tables created in 03_chunk_docs
# delete table with chunked documents
spark.sql(f'DROP TABLE IF EXISTS {destination_tables_config.get("chunked_docs_table_name")}')
# delete (and thus reset) corresponding checkpoint
dbutils.fs.rm(
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('chunked_docs_table_name').split('.')[-1]}",
    True,
)

# delete table with documents failed during chunking
spark.sql(
    f'DROP TABLE IF EXISTS {destination_tables_config.get("chunked_docs_table_name")}_quarantine'
)
# delete (and thus reset) corresponding checkpoint
dbutils.fs.rm(
    f"{destination_tables_config.get('checkpoint_path')}/{destination_tables_config.get('chunked_docs_table_name').split('.')[-1]}_quarantine",
    True,
)

# COMMAND ----------

# DBTITLE 1,Delete vector search index
from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()
try:
    client.delete_index(
        endpoint_name=vectorsearch_config.get("vectorsearch_endpoint_name"),
        index_name=destination_tables_config.get("vectorsearch_index_name"),
    )
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print("Vector Search index doesn't exist, skipping")
        pass  # Handle the case where the resource does not exist, if needed
    else:
        raise e

