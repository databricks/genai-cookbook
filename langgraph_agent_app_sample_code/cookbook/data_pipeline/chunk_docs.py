from typing import Literal, Optional, Any, Callable
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql.functions import explode
import pyspark.sql.functions as func
from typing import Callable
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from pyspark.sql import DataFrame, SparkSession


def apply_chunking_fn(
    parsed_docs_df: DataFrame,
    chunking_fn: Callable[[str], list[str]],
    propagate_columns: list[str],
    doc_column: str = "content",
) -> DataFrame:
    # imports here to avoid requiring these libraries in all notebooks since the data pipeline config imports this package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    import tiktoken

    print(
        f"Applying chunking UDF to {parsed_docs_df.count()} documents using Spark - this may take a long time if you have many documents..."
    )

    parser_udf = func.udf(
        chunking_fn, returnType=ArrayType(StringType()), useArrow=True
    )
    chunked_array_docs = parsed_docs_df.withColumn(
        "content_chunked", parser_udf(doc_column)
    )  # .drop(doc_column)
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

    return chunks_with_ids
