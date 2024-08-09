# Databricks notebook source
# MAGIC %md
# MAGIC #### `chunk_docs`
# MAGIC
# MAGIC `chunk_docs` creates a new delta table, given a table of documents, computing the chunk function over each document to produce a chunked documents table. This utility will let you write the core business logic of the chunker, without dealing with the spark details. You can decide to write your own, or edit this code if it does not fit your use case.
# MAGIC
# MAGIC Arguments:
# MAGIC - `docs_table`: The fully qualified delta table name. For example: `my_catalog.my_schema.my_docs`
# MAGIC - `doc_column`: The name of the column where the documents can be found from `docs_table`. For example: `doc`.
# MAGIC - `chunk_fn`: A function that takes a document (str) and produces a list of chunks (list[str]).
# MAGIC - `propagate_columns`: Columns that should be propagated to the chunk table. For example: `url` to propagate the source URL.
# MAGIC - `chunked_docs_table`: An optional output table name for chunks. Defaults to `{docs_table}_chunked`.
# MAGIC
# MAGIC Returns:
# MAGIC The name of the chunked docs table.
# MAGIC
# MAGIC ##### Examples of creating a `chunk_fn`
# MAGIC
# MAGIC ###### Option 1: Use a recursive character text splitter.
# MAGIC
# MAGIC We provide a `get_recursive_character_text_splitter` util in this cookbook which will determine
# MAGIC the best chunk window given the embedding endpoint that we decide to use for indexing.
# MAGIC
# MAGIC ```py
# MAGIC chunk_fn = get_recursive_character_text_splitter('databricks-bge-large-en')
# MAGIC ```
# MAGIC
# MAGIC ###### Option 2: Use a custom splitter (e.g. LLamaIndex splitters)
# MAGIC
# MAGIC > An example `chunk_fn` using the markdown-aware node parser:
# MAGIC
# MAGIC ```py
# MAGIC from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
# MAGIC from llama_index.core import Document
# MAGIC parser = MarkdownNodeParser()
# MAGIC
# MAGIC def chunk_fn(doc: str) -> list[str]:
# MAGIC   documents = [Document(text=doc)]
# MAGIC   nodes = parser.get_nodes_from_documents(documents)
# MAGIC   return [node.get_content() for node in nodes]
# MAGIC ```

# COMMAND ----------

# If using this in a different context, use:
# %pip install langchain databricks-vectorsearch transformers torch==2.3.0 tiktoken==0.7.0 langchain_core==0.2.5 langchain_community==0.2.4
# dbutils.library.restartPython()

# COMMAND ----------

from typing import Literal, Optional, Any, Callable
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from typing import Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import tiktoken
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType

def compute_chunks(
  docs_table: str,
  doc_column: str,
  chunk_fn: Callable[[str], list[str]],
  propagate_columns: list[str],
  chunked_docs_table: str
) -> str:
  chunked_docs_table = chunked_docs_table or f'{docs_table}_chunked'

  print(f'Computing chunks for `{docs_table}`...')

  raw_docs = spark.read.table(docs_table)

  parser_udf = func.udf(
      chunk_fn,
      returnType=ArrayType(StringType()),
  )
  chunked_array_docs = raw_docs.withColumn("content_chunked", parser_udf(doc_column)).drop(doc_column)
  chunked_docs = chunked_array_docs.select(*propagate_columns, explode("content_chunked").alias("content_chunked"))

  # Add a primary key: "chunk_id".
  chunks_with_ids = chunked_docs.withColumn(
      "chunk_id",
      func.md5(func.col("content_chunked"))
  )
  # Reorder for better display.
  chunks_with_ids = chunks_with_ids.select("chunk_id", "content_chunked", *propagate_columns)

  print(f'Created {chunks_with_ids.count()} chunks!')

  # Write to Delta Table
  chunks_with_ids.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
      chunked_docs_table
  )

  return chunked_docs_table
