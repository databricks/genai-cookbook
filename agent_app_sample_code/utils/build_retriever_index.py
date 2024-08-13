# Databricks notebook source
# MAGIC %md
# MAGIC ##### `build_retriever_index`
# MAGIC
# MAGIC `build_retriever_index` will build the vector search index which is used by our RAG to retrieve relevant documents.
# MAGIC
# MAGIC Arguments:
# MAGIC - `chunked_docs_table`: The chunked documents table. There is expected to be a `chunked_text` column, a `chunk_id` column, and a `url` column.
# MAGIC -  `primary_key`: The column to use for the vector index primary key.
# MAGIC - `embedding_source_column`: The column to compute embeddings for in the vector index.
# MAGIC - `vector_search_endpoint`: An optional vector search endpoint name. It not defined, defaults to the `{table_id}_vector_search`.
# MAGIC - `vector_search_index_name`: An optional index name. If not defined, defaults to `{chunked_docs_table}_index`.
# MAGIC - `embedding_endpoint_name`: An embedding endpoint name.
# MAGIC - `force_delete_vector_search_endpoint`: Setting this to true will rebuild the vector search endpoint.

# COMMAND ----------

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

# COMMAND ----------

# from databricks.sdk import WorkspaceClient

# def create_or_get_vector_search_endpoint(vector_search_endpoint: str):
#   w = WorkspaceClient()
#   vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
#   if sum([vector_search_endpoint == ve.name for ve in vector_search_endpoints]) == 0:
#       print(f"Please wait, creating Vector Search endpoint `{vector_search_endpoint}`.  This can take up to 20 minutes...")
#       w.vector_search_endpoints.create_endpoint_and_wait(vector_search_endpoint, endpoint_type=EndpointType.STANDARD)

#   # Make sure vector search endpoint is online and ready.
#   w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(vector_search_endpoint)

#   print(f"PASS: Vector Search endpoint `{vector_search_endpoint}` exists")

# COMMAND ----------


from typing import TypedDict, Dict
import io 
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from mlflow.utils import databricks_utils as du
from functools import partial
import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from databricks.vector_search.client import VectorSearchClient
import mlflow

def _build_index(
    primary_key: str,
    embedding_source_column: str,
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vectorsearch_index_name: str,
    embedding_endpoint_name: str,
    force_delete=False):

  # Get the vector search index
  vsc = VectorSearchClient(disable_notice=True)

  # Use optimizations if available
#   dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
#   if dbr_majorversion >= 14:
#     spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

  def find_index(endpoint_name, index_name):
      all_indexes = vsc.list_indexes(name=vector_search_endpoint).get("vector_indexes", [])
      return vectorsearch_index_name in map(lambda i: i.get("name"), all_indexes)

  if find_index(endpoint_name=vector_search_endpoint, index_name=vectorsearch_index_name):
      if force_delete:
          vsc.delete_index(endpoint_name=vector_search_endpoint, index_name=vectorsearch_index_name)
          create_index = True
      else:
          create_index = False
          # sync the index
          print("Syncing index, this can take 15 minutes or much longer if you have a larger number of documents...")
          print(f'Check status at: {get_table_url(vectorsearch_index_name)}')
          
          sync_result = vsc.get_index(index_name=vectorsearch_index_name).sync()

  else:
      print(f'Creating non-existent vector search index for endpoint "{vector_search_endpoint}" and index "{vectorsearch_index_name}"')
      create_index = True

  if create_index:
      print("Computing document embeddings and Vector Search Index. This can take 15 minutes or much longer if you have a larger number of documents.")
      print(f'Check status at: {get_table_url(vectorsearch_index_name)}')

      vsc.create_delta_sync_index_and_wait(
          endpoint_name=vector_search_endpoint,
          index_name=vectorsearch_index_name,
          primary_key=primary_key, #"chunk_id",
          source_table_name=chunked_docs_table_name,
          pipeline_type="TRIGGERED",
          embedding_source_column=embedding_source_column, #"chunked_text",
          embedding_model_endpoint_name=embedding_endpoint_name
      )

# #   tag_delta_table(vectorsearch_index_name, data_pipeline_config)
#   mlflow.log_input(mlflow.data.load_delta(table_name=chunked_docs_table_name), context="chunked_docs")

# COMMAND ----------

from pydantic import BaseModel

class RetrieverIndexResult(BaseModel):
  vector_search_endpoint: str
  vector_search_index_name: str
  embedding_endpoint_name: str
  chunked_docs_table: str

def build_retriever_index(
  chunked_docs_table: str,
  primary_key: str,
  embedding_source_column: str,
  embedding_endpoint_name: str,
  vector_search_endpoint: str,
  vector_search_index_name: str,
  force_delete_vector_search_endpoint=False) -> RetrieverIndexResult:
  
  retriever_index_result = RetrieverIndexResult(
    # TODO(nsthorat): Is this right? Should we make a new vector search index for each chunked docs table?
    # TODO(e): No it will be many indexes per endpoint
    vector_search_endpoint=vector_search_endpoint,
    vector_search_index_name=vector_search_index_name,
    embedding_endpoint_name=embedding_endpoint_name,
    chunked_docs_table=chunked_docs_table
  )

  # Enable CDC for Vector Search Delta Sync
  spark.sql(
    f"ALTER TABLE {chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
  )

  print('Building embedding index...')
  # Building the index.
  _build_index(
      primary_key=primary_key,
      embedding_source_column=embedding_source_column,
      vector_search_endpoint=vector_search_endpoint,
      chunked_docs_table_name=chunked_docs_table,
      vectorsearch_index_name=vector_search_index_name,
      embedding_endpoint_name=embedding_endpoint_name,
      force_delete=force_delete_vector_search_endpoint)
  
 

  return retriever_index_result
