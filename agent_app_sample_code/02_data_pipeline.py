# Databricks notebook source
# MAGIC %md
# MAGIC # Unstructured data pipeline for the Agent's Retriever
# MAGIC
# MAGIC By the end of this notebook, you will have transformed your unstructured documents into a format that can be queried by your Agent.
# MAGIC
# MAGIC This means:
# MAGIC - Documents loaded into a delta table.
# MAGIC - Documents are chunked.
# MAGIC - Chunks have been embedded with an embedding model and stored in a vector index.
# MAGIC
# MAGIC The important resulting artifact of this notebook is the chunked vector index. This will be used in the next notebook to power our RAG application.
# MAGIC
# MAGIC After you have initial feedback from your stakeholders, you can easily adapt and tweak this pipeline to fit more advanced logic. For example, if your retrieval quality is low, you could experiment with different parsing and chunk sizes once you gain working knowledge of your data.
# MAGIC
# MAGIC This notebook includes smart defaults for parsing, chunking, and embedding model.  The notebook is designed such that can be easily modified based on your data e.g., customize the parsing or chunking logic, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install necessary libraries for parsing

# COMMAND ----------

# Packages required by all code.
# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
# Versions of open source packages are locked since package authors often make backwards compatible changes
%pip install -qqqq -U \
  databricks-vectorsearch databricks-agents pydantic databricks-sdk mlflow mlflow-skinny `# For agent & data pipeline code` \
  pypdf==4.1.0  `# PDF parsing` \
  markdownify==0.12.1  `# HTML parsing` \
  pypandoc_binary==1.13  `# DOCX parsing` \
  transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0. `# get_recursive_character_text_splitter`

# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the global configuration

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiment name
# MAGIC
# MAGIC Used to track information about this Data Pipeline that are used in the later notebooks.

# COMMAND ----------

import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data pipeline configuration
# MAGIC
# MAGIC - `VECTOR_SEARCH_ENDPOINT`: [Create a Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to host the resulting vector index
# MAGIC - `SOURCE_PATH`: A [UC Volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html#create-and-work-with-volumes) that contains the source documents for your Agent.
# MAGIC - `EMBEDDING_MODEL_ENDPOINT`: Model Serving endpoint - either an [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) Provisioned Throughput / Pay-Per-Token or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/embeddings`/.  Supported models:
# MAGIC     - FMAPI `databricks-gte-large-en` or `databricks-bge-large-en`
# MAGIC     - Provisioned Throughput `databricks-gte-large-en` or `databricks-bge-large-en`
# MAGIC     - Azure OpenAI or OpenAI External Model of type `text-embedding-ada-002`, `text-embedding-3-small` or `text-embedding-3-large`

# COMMAND ----------

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f"{user_name}_vector_search"

# Source location for documents
# You need to create this location and add files
SOURCE_UC_VOLUME = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/source_docs"

# Names of the output Delta Tables tables & Vector Search index

# Delta Table with the parsed documents and extracted metadata
DOCS_DELTA_TABLE = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_docs`"

# Chunked documents that are loaded into the Vector Index
CHUNKED_DOCS_DELTA_TABLE = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_chunked_docs`"

# Vector Index
VECTOR_INDEX_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}_chunked_docs_index"

# Embedding model endpoint. The list of off-the-shelf embeddings can be found here:
# https://docs.databricks.com/en/machine-learning/foundation-models/index.html
EMBEDDING_MODEL_ENDPOINT = "databricks-bge-large-en"
# EMBEDDING_MODEL_ENDPOINT = "ep-embeddings-small"
# EMBEDDING_MODEL_ENDPOINT = "bge-test"

# COMMAND ----------

# MAGIC %md ## Validate config & create resources
# MAGIC
# MAGIC This step will check your configuration and create UC catalog/schema + the Vector Search endpoint if they do not exist.

# COMMAND ----------

# MAGIC %run ./validators/validate_data_pipeline_config

# COMMAND ----------

create_or_check_volume_path(SOURCE_UC_VOLUME)
create_or_check_vector_search_endpoint(VECTOR_SEARCH_ENDPOINT)
validate_embedding_endpoint(
    endpoint_name=EMBEDDING_MODEL_ENDPOINT, task_type="llm/v1/embeddings"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Pipeline 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start MLflow run for tracking

# COMMAND ----------

mlflow.end_run()
mlflow.start_run(run_name=POC_DATA_PIPELINE_RUN_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load documents into a delta table
# MAGIC
# MAGIC In this step, we'll load files from `SOURCE_PATH` (defined in your `00_global_config`) into a delta table. The contents of each file will become a separate row in our delta table.
# MAGIC
# MAGIC The path to the source document will be used as the `doc_uri` which is displayed to your end users in the Agent's web application.
# MAGIC
# MAGIC After you test your POC with stakeholders, you can return here to change the parsing logic or extraction additional metadata about the documents to help improve the quality of your retriever.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define your parsing function
# MAGIC
# MAGIC This default implementation parses PDF, HTML, and DOCX files using open source libraries.  Adjust `file_parser(...)` to add change the parsing logic, add support for more file types, or extract additional metadata about each document.

# COMMAND ----------

from typing import TypedDict
from datetime import datetime
import warnings
import io
import traceback
import os
from urllib.parse import urlparse

# PDF libraries
from pypdf import PdfReader

# HTML libraries
from markdownify import markdownify as md
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

# Schema of the dict returned by `file_parser(...)`
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES - these are required by Evaluation & Framework
    # Parsed content of the document
    doc_content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name

    # OK TO CHANGE THESE NAMES
    # Optionally, you can add additional metadata fields here
    example_metadata: str
    last_modified: datetime


# Parser function.  Replace this function to provide custom parsing logic.
def file_parser(
    raw_doc_contents_bytes: bytes,
    doc_path: str,
    modification_time: datetime,
    doc_bytes_length: int,
) -> ParserReturnValue:
    """
    Parses the content of a PDF document into a string.

    This function takes the raw bytes of a PDF document and its path, attempts to parse the document using PyPDF,
    and returns the parsed content and the status of the parsing operation.

    Parameters:
    - raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed (set by Spark when loading the file)
    - doc_path (str): The DBFS path of the document, used to verify the file extension (set by Spark when loading the file)
    - modification_time (timestamp): The last modification time of the document (set by Spark when loading the file)
    - doc_bytes_length (long): The size of the document in bytes (set by Spark when loading the file)

    Returns:
    - ParserReturnValue: A dictionary containing the parsed document content and the status of the parsing operation.
      The 'doc_content' key will contain the parsed text as a string, and the 'parser_status' key will indicate
      whether the parsing was successful or if an error occurred.
    """
    try:
        filename, file_extension = os.path.splitext(doc_path)
        parsed_document = {}

        if file_extension == ".pdf":
            pdf = io.BytesIO(raw_doc_contents_bytes)
            reader = PdfReader(pdf)

            parsed_content = [
                page_content.extract_text() for page_content in reader.pages
            ]

            parsed_document = {
                "doc_content": "\n".join(parsed_content),
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".html":
            from markdownify import markdownify as md

            html_content = raw_doc_contents_bytes.decode("utf-8")

            markdown_contents = md(
                str(html_content).strip(), heading_style=markdownify.ATX
            )
            markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

            parsed_document = {
                "doc_content": markdown_stripped,
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".docx":
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(raw_doc_contents_bytes)
                temp_file_path = temp_file.name
                md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

                parsed_document = {
                    "doc_content": md.strip(),
                    "parser_status": "SUCCESS",
                }
        else:
            raise Exception(f"No supported parser for {doc_path}")

        # Extract the required doc_uri
        # convert from `dbfs:/Volumes/catalog/schema/pdf_docs/filename.pdf` to `Volumes/catalog/schema/pdf_docs/filename.pdf`
        modified_path = urlparse(doc_path).path.lstrip('/')
        parsed_document["doc_uri"] = modified_path

        # Sample metadata extraction logic
        if "test" in parsed_document["doc_content"]:
            parsed_document["example_metadata"] = "test"
        else:
            parsed_document["example_metadata"] = "not test"

        # Add the modified time
        parsed_document["last_modified"] = modification_time

        return parsed_document

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "doc_content": "",
            "parser_status": f"ERROR: {status}",
        }

# COMMAND ----------

# MAGIC %run ./utils/typed_dicts_to_spark_schema

# COMMAND ----------

from utils.file_loading import load_files_to_df, apply_parsing_udf

# COMMAND ----------

raw_files_df = load_files_to_df(
    spark=spark,
    source_path=SOURCE_UC_VOLUME,
)

parsed_files_df = apply_parsing_udf(
    raw_files_df=raw_files_df,
    # Modify this function to change the parser, extract additional metadata, etc
    parse_file_udf=file_parser,
    # The schema of the resulting Delta Table will follow the schema defined in ParserReturnValue
    parsed_df_schema=typed_dicts_to_spark_schema(ParserReturnValue)
)

# Write to a Delta Table
parsed_files_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(DOCS_DELTA_TABLE)

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")
parsed_files_df.display()

# Log the resulting table to MLflow
mlflow.log_input(
    mlflow.data.load_delta(
        table_name=DOCS_DELTA_TABLE, name=DOCS_DELTA_TABLE.replace("`", "")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunks of documents
# MAGIC
# MAGIC Now we will split our documents into smaller chunks so they can be indexed in our vector database.
# MAGIC
# MAGIC We've provided a utility `chunk_docs` to help with this. Alternatively, you could compute chunks in your own spark pipeline and pass it to the indexing step after chunking.
# MAGIC

# COMMAND ----------

# MAGIC %run ./utils/chunk_docs

# COMMAND ----------

# MAGIC %run ./utils/get_recursive_character_text_splitter

# COMMAND ----------

# Configure the chunker
chunk_fn = get_recursive_character_text_splitter(
    model_serving_endpoint=EMBEDDING_MODEL_ENDPOINT,
    chunk_size_tokens=384,
    chunk_overlap_tokens=128,
)

# Get the columns from the parser except for the doc_content
# You can modify this to adjust which fields are propagated from the docs table to the chunks table.
propagate_columns = [
    field.name
    for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
    if field.name != "doc_content"
]

chunked_docs_table = compute_chunks(
    # The source documents table.
    docs_table=DOCS_DELTA_TABLE,
    # The column containing the documents to be chunked.
    doc_column="doc_content",
    # The chunking function that takes a string (document) and returns a list of strings (chunks).
    chunk_fn=chunk_fn,
    # Choose which columns to propagate from the docs table to chunks table. `doc_uri` column is required we can propagate the original document URL to the Agent's web app.
    propagate_columns=propagate_columns,
    # By default, the chunked_docs_table will be written to `{docs_table}_chunked`.
    chunked_docs_table=CHUNKED_DOCS_DELTA_TABLE,
)

display(spark.read.table(chunked_docs_table))

# Log to MLflow
mlflow.log_input(
    mlflow.data.load_delta(
        table_name=CHUNKED_DOCS_DELTA_TABLE,
        name=CHUNKED_DOCS_DELTA_TABLE.replace("`", ""),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the vector index
# MAGIC
# MAGIC Next, we'll compute the vector index over the chunks and create our retriever index that will be used to query relevant documents to the user question.

# COMMAND ----------

# MAGIC %run ./utils/build_retriever_index

# COMMAND ----------

retriever_index_result = build_retriever_index(
    # Spark requires `` to escape names with special chars, VS client does not.
    chunked_docs_table=CHUNKED_DOCS_DELTA_TABLE.replace("`", ""),
    primary_key="chunk_id",
    embedding_source_column="content_chunked",
    vector_search_endpoint=VECTOR_SEARCH_ENDPOINT,
    vector_search_index_name=VECTOR_INDEX_NAME,
    # Must match the embedding endpoint you used to chunk your documents
    embedding_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
    # Set to true to re-create the vector search endpoint when re-running.
    force_delete_vector_search_endpoint=False,
)

print(retriever_index_result)

print()
print("Vector search index created! This will be used in the next notebook.")
print(f"Vector search endpoint: {retriever_index_result.vector_search_endpoint}")
print(f"Vector search index: {retriever_index_result.vector_search_index_name}")
print(f"Embedding used: {retriever_index_result.embedding_endpoint_name}")
print(f"Chunked docs table: {retriever_index_result.chunked_docs_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## End Mlflow run

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


