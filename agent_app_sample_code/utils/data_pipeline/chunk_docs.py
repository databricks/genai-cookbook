from typing import Literal, Optional, Any, Callable
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from typing import Callable
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from pyspark.sql import DataFrame, SparkSession



# %md
# #### `chunk_docs`

# `chunk_docs` creates a new delta table, given a table of documents, computing the chunk function over each document to produce a chunked documents table. This utility will let you write the core business logic of the chunker, without dealing with the spark details. You can decide to write your own, or edit this code if it does not fit your use case.

# Arguments:
# - `docs_table`: The fully qualified delta table name. For example: `my_catalog.my_schema.my_docs`
# - `doc_column`: The name of the column where the documents can be found from `docs_table`. For example: `doc`.
# - `chunk_fn`: A function that takes a document (str) and produces a list of chunks (list[str]).
# - `propagate_columns`: Columns that should be propagated to the chunk table. For example: `url` to propagate the source URL.
# - `chunked_docs_table`: An optional output table name for chunks. Defaults to `{docs_table}_chunked`.

# Returns:
# The name of the chunked docs table.

# ##### Examples of creating a `chunk_fn`

# ###### Option 1: Use a recursive character text splitter.

# We provide a `get_recursive_character_text_splitter` util in this cookbook which will determine
# the best chunk window given the embedding endpoint that we decide to use for indexing.

# ```py
# chunk_fn = get_recursive_character_text_splitter('databricks-bge-large-en')
# ```

# ###### Option 2: Use a custom splitter (e.g. LLamaIndex splitters)

# > An example `chunk_fn` using the markdown-aware node parser:

# ```py
# from llama_index.core.node_parser import MarkdownNodeParser, TokenTextSplitter
# from llama_index.core import Document
# parser = MarkdownNodeParser()

# def chunk_fn(doc: str) -> list[str]:
#   documents = [Document(text=doc)]
#   nodes = parser.get_nodes_from_documents(documents)
#   return [node.get_content() for node in nodes]
# ```



def compute_chunks(
    docs_table: str,
    doc_column: str,
    chunk_fn: Callable[[str], list[str]],
    propagate_columns: list[str],
    chunked_docs_table: str,
    spark: SparkSession
) -> str:
    # imports here to avoid requiring these libraries in all notebooks since the data pipeline config imports this package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    import tiktoken

    chunked_docs_table = chunked_docs_table or f"{docs_table}_chunked"

    print(f"Computing chunks for `{docs_table}`...")

    raw_docs = spark.read.table(docs_table)

    parser_udf = func.udf(
        chunk_fn,
        returnType=ArrayType(StringType()),
    )
    chunked_array_docs = raw_docs.withColumn(
        "content_chunked", parser_udf(doc_column)
    )#.drop(doc_column)
    chunked_docs = chunked_array_docs.select(
        *propagate_columns, explode("content_chunked").alias("content_chunked")
    )

    # Add a primary key: "chunk_id".
    chunks_with_ids = chunked_docs.withColumn(
        "chunk_id", func.md5(func.col("content_chunked"))
    )
    # Reorder for better display.
    chunks_with_ids = chunks_with_ids.select(
        "chunk_id", "content_chunked", *propagate_columns
    )

    print(f"Created {chunks_with_ids.count()} chunks!")

    # Write to Delta Table
    chunks_with_ids.write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(chunked_docs_table)

    # enable CDC feed for VS index sync
    spark.sql(f"ALTER TABLE {chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    return chunked_docs_table