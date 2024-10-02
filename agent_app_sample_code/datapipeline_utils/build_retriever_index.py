from typing import TypedDict, Dict
import io
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
from mlflow.utils import databricks_utils as du
from functools import partial
from databricks.vector_search.client import VectorSearchClient
import mlflow



# %md
# ##### `build_retriever_index`

# `build_retriever_index` will build the vector search index which is used by our RAG to retrieve relevant documents.

# Arguments:
# - `chunked_docs_table`: The chunked documents table. There is expected to be a `chunked_text` column, a `chunk_id` column, and a `url` column.
# -  `primary_key`: The column to use for the vector index primary key.
# - `embedding_source_column`: The column to compute embeddings for in the vector index.
# - `vector_search_endpoint`: An optional vector search endpoint name. It not defined, defaults to the `{table_id}_vector_search`.
# - `vector_search_index_name`: An optional index name. If not defined, defaults to `{chunked_docs_table}_index`.
# - `embedding_endpoint_name`: An embedding endpoint name.
# - `force_delete_vector_search_endpoint`: Setting this to true will rebuild the vector search endpoint.



def build_retriever_index(
    primary_key: str,
    embedding_source_column: str,
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vector_search_index_name: str,
    embedding_endpoint_name: str,
    force_delete_index_before_create=False,
):

    # Get the vector search index
    vsc = VectorSearchClient(disable_notice=True)

    def find_index(index_name):
        try:
            vsc.get_index(index_name=index_name)
            return True
        except Exception as e:
            return False


    if find_index(
        index_name=vector_search_index_name
    ):
        if force_delete_index_before_create:
            vsc.delete_index(
                 index_name=vector_search_index_name
            )
            create_index = True
        else:
            create_index = False
            print(
                f"Syncing index {vector_search_index_name}, this can take 15 minutes or much longer if you have a larger number of documents..."
            )

            sync_result = vsc.get_index(index_name=vector_search_index_name).sync()

    else:
        print(
            f'Creating non-existent vector search index for endpoint "{vector_search_endpoint}" and index "{vector_search_index_name}"'
        )
        create_index = True

    if create_index:
        print(
            f"Computing document embeddings and Vector Search Index. This can take 15 minutes or much longer if you have a larger number of documents."
        )
        try:
            vsc.create_delta_sync_index_and_wait(
                endpoint_name=vector_search_endpoint,
                index_name=vector_search_index_name,
                primary_key=primary_key,
                source_table_name=chunked_docs_table_name,
                pipeline_type="TRIGGERED",
                embedding_source_column=embedding_source_column,
                embedding_model_endpoint_name=embedding_endpoint_name,
            )
            print("SUCCESS: Vector search index created.")
        except Exception as e:
            print(f"\n\nERROR: Vector search index creation failed. {e}.\n\nWait 5 minutes and try running this cell again.")

        