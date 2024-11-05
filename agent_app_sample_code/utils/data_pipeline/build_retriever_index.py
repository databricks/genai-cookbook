from databricks.sdk.service.vectorsearch import (
    VectorSearchIndexesAPI,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist, BadRequest
import time
from utils.cookbook.databricks_utils import get_table_url


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
    vector_search_endpoint: str,
    chunked_docs_table_name: str,
    vector_search_index_name: str,
    embedding_endpoint_name: str,
    force_delete_index_before_create=False,
    primary_key: str = "chunk_id",  # hard coded in the apply_chunking_fn
    embedding_source_column: str = "content_chunked",  # hard coded in the apply_chunking_fn
) -> tuple[bool, str]:
    # Initialize workspace client and vector search API
    w = WorkspaceClient()
    vsc = w.vector_search_indexes

    def find_index(index_name):
        try:
            return vsc.get_index(index_name=index_name)
        except ResourceDoesNotExist:
            return None

    def wait_for_index_to_be_ready(index):
        while not index.status.ready:
            print(
                f"Index {vector_search_index_name} exists, but is not ready, waiting 30 seconds..."
            )
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    def wait_for_index_to_be_deleted(index):
        while index:
            print(
                f"Waiting for index {vector_search_index_name} to be deleted, waiting 30 seconds..."
            )
            time.sleep(30)
            index = find_index(index_name=vector_search_index_name)

    existing_index = find_index(index_name=vector_search_index_name)
    if existing_index:
        print(f"Found existing index {get_table_url(vector_search_index_name)}...")
        if force_delete_index_before_create:
            print(f"Deleting index {vector_search_index_name}...")
            vsc.delete_index(index_name=vector_search_index_name)
            wait_for_index_to_be_deleted(existing_index)
            create_index = True
        else:
            wait_for_index_to_be_ready(existing_index)
            create_index = False
            print(
                f"Starting the sync of index {vector_search_index_name}, this can take 15 minutes or much longer if you have a larger number of documents."
            )
            # print(existing_index)
            try:
                vsc.sync_index(index_name=vector_search_index_name)
                msg = f"Kicked off index sync for {vector_search_index_name}."
                return (False, msg)
            except BadRequest as e:
                msg = f"Index sync already in progress, so failed to kick off index sync for {vector_search_index_name}.  Please wait for the index to finish syncing and try again."
                return (True, msg)
    else:
        print(
            f'Creating new vector search index "{vector_search_index_name}" on endpoint "{vector_search_endpoint}"'
        )
        create_index = True

    if create_index:
        print(
            "Computing document embeddings and Vector Search Index. This can take 15 minutes or much longer if you have a larger number of documents."
        )
        try:
            # Create delta sync index spec using the proper class
            delta_sync_spec = DeltaSyncVectorIndexSpecRequest(
                source_table=chunked_docs_table_name,
                pipeline_type=PipelineType.TRIGGERED,
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name=embedding_source_column,
                        embedding_model_endpoint_name=embedding_endpoint_name,
                    )
                ],
            )

            vsc.create_index(
                name=vector_search_index_name,
                endpoint_name=vector_search_endpoint,
                primary_key=primary_key,
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=delta_sync_spec,
            )
            msg = (
                f"Successfully created vector search index {vector_search_index_name}."
            )
            print(msg)
            return (False, msg)
        except Exception as e:
            msg = f"Vector search index creation failed. Wait 5 minutes and try running this cell again."
            return (True, msg)
