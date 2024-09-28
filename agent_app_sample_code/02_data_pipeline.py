# Databricks notebook source
# MAGIC %md
# MAGIC # Unstructured data pipeline for the Agent's Retriever
# MAGIC
# MAGIC By the end of this notebook, you will have transformed your unstructured documents into a vector index that can be queried by your Agent.
# MAGIC
# MAGIC This means:
# MAGIC - Documents loaded into a delta table.
# MAGIC - Documents are chunked.
# MAGIC - Chunks have been embedded with an embedding model and stored in a vector index.
# MAGIC
# MAGIC The important resulting artifact of this notebook is the chunked vector index. This will be used in the next notebook to power our Retriever.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC We suggest the following approach to using this notebook to build and iterate on your data pipeline's quality.
# MAGIC 1. **Build the first version of your index with our smart default settings**
# MAGIC     - Install the Python libraries
# MAGIC     - 1️⃣ 📂 Configure data source & destination tables 
# MAGIC     - Press Run All to create the vector index
# MAGIC
# MAGIC     *Note: For your initial data pipeline, this notebook is designed so you can **only** adjust the data source/destinaton configuration and then press run all.*
# MAGIC
# MAGIC 2. **Use the later notebooks to deploy the Agent and use Agent Evaluation to measure the quality of your Agent + Retriever.**
# MAGIC 3. **If your evaluation results indicate a retrieval issue, try various strategies to improve the quality.  For a deep dive on these strategies, view [AI cookbook](https://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-1-retrieval.html).**
# MAGIC     - Verify that the necessary source documents are included
# MAGIC     - Resolve conflicting source documents
# MAGIC     - 2️⃣ ⚙️ Adjust the data pipeline config
# MAGIC       - Change the chunk size or overlap
# MAGIC       - Try different embedding models
# MAGIC     - 3️⃣ ⌨️ Adjust the data pipeline code
# MAGIC       - Write a custom parser or try different parsing libraries
# MAGIC       - Write a custom chunker or try different chunking techniques
# MAGIC       - Extract additional metadata about each document
# MAGIC     - Adjust the Agent's code/config *(this is done the following notebooks)*
# MAGIC       - Change the K e.g., number of docs retrieved
# MAGIC       - Try a re-ranker
# MAGIC       - Try hybrid search
# MAGIC       - Use extracted metadata as filters

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Python libraries
# MAGIC
# MAGIC You do not need to modify this cell unless you are adjusting the code for the document parsing or chunking logic.

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
# Versions of open source packages are locked since package authors often make backwards compatible changes
%pip install -qqqq -U \
  -r requirements.txt `# Packages shared across all notebooks` \
  pymupdf4llm==0.0.5 pymupdf==1.24.5 `# PDF parsing` \
  markdownify==0.12.1  `# HTML parsing` \
  pypandoc_binary==1.13  `# DOCX parsing` \
  transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0. `# For get_recursive_character_text_splitter`

# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1️⃣ 📂 Data source & destination configuration

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📂 (Optional) Change Agent's shared storage location configuration
# MAGIC
# MAGIC **❗❗ If you configured `00_shared_config`, just run this cell as-is.**
# MAGIC
# MAGIC From the shared configuration, we use:
# MAGIC * The UC catalog & schema as the default location for output tables/indexs
# MAGIC * The MLflow experiment for tracking pipeline runs
# MAGIC
# MAGIC *These are set in `00_shared_config`, but can also be overriden here if you want to use this notebook independently.*
# MAGIC
# MAGIC

# COMMAND ----------

# %load_ext autoreload

# %autoreload 2

# COMMAND ----------

from utils import AgentStorageLocationConfig
import mlflow

# Load the shared configuration
agent_storage_location_config = AgentStorageLocationConfig.from_yaml_file('./configs/agent_config.yaml')

# Print configuration 
agent_storage_location_config.pretty_print()

# Set the MLflow Experiment that is used to track metadata about each run of this Data Pipeline.
experiment_info = mlflow.set_experiment(agent_storage_location_config.mlflow_experiment_directory)

# COMMAND ----------

# MAGIC %md #### 📂 Configure the data pipeline's source location.
# MAGIC
# MAGIC Choose a [Unity Catalog Volume](https://docs.databricks.com/en/volumes/index.html) containing PDF, HTML, etc documents to be parsed/chunked/embedded.
# MAGIC
# MAGIC - `uc_catalog_name`: Name of the Unity Catalog.
# MAGIC - `uc_schema_name`: Name of the Unity Catalog schema.
# MAGIC - `uc_volume_name`: Name of the Unity Catalog volume. 
# MAGIC
# MAGIC Running this cell with validate that the UC Volume exists, trying to create it if not.

# COMMAND ----------

from cookbook_utils import UnstructuredDataPipelineSourceConfig

# Default is a Volume called `{uc_asset_prefix}_source_docs` in the configured UC catalog/schema
source_config = UnstructuredDataPipelineSourceConfig(
    uc_catalog_name=agent_storage_location_config.uc_catalog,
    uc_schema_name=agent_storage_location_config.uc_schema,
    uc_volume_name=f"{agent_storage_location_config.uc_asset_prefix}_source_docs"
)

# Print source location to console
source_config.pretty_print()

# Save to reference in later notebooks
source_config.dump_to_yaml("./configs/data_pipeline_source_config.yaml")

# Check if volume exists, create otherwise
if not source_config.create_or_check_volume():
  raise Exception("UC Volume is not valid, fix per the console notes above.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 📂 Configure the data pipeline's output locations.
# MAGIC
# MAGIC Choose where the data pipeline outputs the parsed, chunked, and embedded documents.
# MAGIC
# MAGIC By default, the Delta Tables and Vector Index are created in the `agent_storage_location_config`'s Unity Catalog schema and given a logical name e.g., `catalog.schema.{uc_asset_prefix}_docs`.  Optionally, you can change the name of the tables/indexes.  
# MAGIC
# MAGIC *Note: If you are comparing different chunking/parsing/embedding strategies, set the `tag` parameter, which is appended to the output tables/indexes so you can have multiple Vector Indexes running at once.*
# MAGIC
# MAGIC * `parsed_docs_table`: Delta Table to store parsed documents.
# MAGIC * `chunked_docs_table`: Delta Table to store chunks of the parsed documents. 
# MAGIC * `vector_index_name`: Vector Search index that is created from `chunked_docs_table`. 
# MAGIC * `vector_search_endpoint`: Vector Search endpoint to store the index.
# MAGIC
# MAGIC *Databricks suggests sharing a Vector Search endpoint across multiple agents.  To do so, replace the default value for `vector_search_endpoint` with an existing endpoint.*

# COMMAND ----------

from cookbook_utils import UnstructuredDataPipelineStorageConfig
from databricks.sdk import WorkspaceClient

uc_asset_prefix = agent_storage_location_config.uc_asset_prefix

storage_config = UnstructuredDataPipelineStorageConfig(
    uc_catalog_name=agent_storage_location_config.uc_catalog,
    uc_schema_name=agent_storage_location_config.uc_schema,
    uc_asset_prefix=uc_asset_prefix,
    parsed_docs_table=f"{uc_asset_prefix}_docs",
    chunked_docs_table=f"{uc_asset_prefix}_docs_chunked",
    vector_index=f"{uc_asset_prefix}_docs_chunked_index",
    # vector_search_endpoint=f"{uc_asset_prefix}_endpoint", # by default, a new endpoint is created for the Agent
    vector_search_endpoint="ericpeter_vector_search",
    tag="22", # Optional, use to tag the tables/index with a postfix to differentiate between versions when iterating on chunking/parsing/embedding configs.  
)

# Print output locations to console
storage_config.pretty_print()

# Save to reference in later notebooks
storage_config.dump_to_yaml("./configs/data_pipeline_storage_config.yaml")

# Verify Vector Search endpoint, create if it does not exist
if not storage_config.create_or_check_vector_search_endpoint():
  raise Exception("Vector Search endpoint is not valid, fix per the console notes above.")

# COMMAND ----------

# MAGIC %md #### 🛑 You can stop here if you are running your initial data pipeline.  You will come back to these steps later to tune the quality of your data pipeline.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2️⃣ ⚙️ Adjust the data pipeline config

# COMMAND ----------

# MAGIC %md
# MAGIC #### ⚙️ Configure simple chunking parameters and the embedding model.
# MAGIC
# MAGIC **Chunk size and overlap** control how a larger document is turned into smaller chunks that can be processed by an embedding model.  See the AI Cookbook [chunking deep dive](https://ai-cookbook.io/nbs/3-deep-dive-data-pipeline.html#chunking) for more details.
# MAGIC
# MAGIC **The embedding model** is an AI model that is used to identify the most similar documents to a given user's query.  See the AI Cookbook [embedding model deep dive](https://ai-cookbook.io/nbs/3-deep-dive-data-pipeline.html#embedding-model) for more details.
# MAGIC
# MAGIC This notebook supports the following [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/embeddings`/.  If you want to try another model, you will need to modify the `utils/get_recursive_character_text_splitter` Notebook to add support.
# MAGIC - `databricks-gte-large-en` or `databricks-bge-large-en`
# MAGIC - Azure OpenAI or OpenAI External Model of type `text-embedding-ada-002`, `text-embedding-3-small` or `text-embedding-3-large`

# COMMAND ----------

from cookbook_utils import ChunkingConfig

chunking_config = ChunkingConfig(
    embedding_model_endpoint="databricks-gte-large-en", # A Model Serving endpoint
    chunk_size_tokens=2048,
    chunk_overlap_tokens=256
)

# Print config to console
chunking_config.pretty_print()

# Save to reference in later notebooks
chunking_config.dump_to_yaml("./configs/data_pipeline_chunking_config.yaml")

if not chunking_config.validate_embedding_endpoint():
  raise Exception("`embedding_model_endpoint` is not valid, fix per the console notes above.")

if not chunking_config.validate_chunk_size_and_overlap():
  raise Exception("`chunk_size_tokens` and `chunk_overlap_tokens` is not valid, fix per the console notes above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ ⌨️ Data pipeline code
# MAGIC
# MAGIC The code below executes the data pipeline.  You can modify the below code as indicated to implement different parsing or chunking strategies or to extract additional metadata fields
# MAGIC
# MAGIC Throughout this section, we indicate which cell's code you:
# MAGIC - ✏️ should customize - these cells contain the "business logic" that parses/chunks the documents.
# MAGIC - 🚫 should not customize - these cells contain boilerplate logic to execute the parsing/chunking/embedding inside Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫 Start MLflow run for tracking

# COMMAND ----------

mlflow.end_run()

# This tag appears in the MLflow UI so you can easily identify this run
mlflow_run_tag = storage_config.tag if storage_config.tag is not None or len(storage_config.tag) > 0 else "initial-data-pipeline"
mlflow.start_run(run_name=mlflow_run_tag)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline step 1: Load & parse documents into a Delta Table
# MAGIC
# MAGIC In this step, we'll load files from the UC Volume defined in `source_config` into the Delta Table `storage_config.parsed_docs_table` . The contents of each file will become a separate row in our delta table.
# MAGIC
# MAGIC The path to the source document will be used as the `doc_uri` which is displayed to your end users in the Agent Evalution web application.
# MAGIC
# MAGIC After you test your POC with stakeholders, you can return here to change the parsing logic or extraction additional metadata about the documents to help improve the quality of your retriever.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Customize the parsing function
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
# from pypdf import PdfReader
import fitz
import pymupdf4llm

# HTML libraries
from markdownify import markdownify as md
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

# Schema of the dict returned by `file_parser(...)`
# This is used to create the output Delta Table's schema - if you want to add columns, do so here.
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES - these are required by Agent Evaluation & Framework
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

        if file_extension == ".pdf":
            # pdf = io.BytesIO(raw_doc_contents_bytes)
            # reader = PdfReader(pdf)

            # parsed_content = [
            #     page_content.extract_text() for page_content in reader.pages
            # ]

            pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(pdf_doc)

            parsed_document = {
                "doc_content": md_text.strip(),
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
        modified_path = urlparse(doc_path).path
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

# MAGIC %md 🚫 The below cell is boilerplate code to apply the parsing function using Spark.  You should not need to modify this code.

# COMMAND ----------

from cookbook_utils.file_loading import load_files_to_df, apply_parsing_udf
from cookbook_utils import typed_dicts_to_spark_schema

raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_config.volume_path,
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
).saveAsTable(storage_config.parsed_docs_table)

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")
parsed_files_df.display()

# Log the resulting table to MLflow
mlflow.log_input(
    mlflow.data.load_delta(
        table_name=storage_config.parsed_docs_table, name=storage_config.parsed_docs_table.replace("`", "")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline step 2: Compute chunks of documents
# MAGIC
# MAGIC In this step, we will split our documents into smaller chunks so they can be indexed in our vector database.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ✏️ Chunking logic.  We provide a default implementation of a recursive text splitter.  To create your own chunking logic, adapt the `get_recursive_character_text_splitter()` function inside `cookbook_utils.recursive_character_text_splitter.py`.

# COMMAND ----------

from cookbook_utils.recursive_character_text_splitter import get_recursive_character_text_splitter

# Configure the chunker
chunk_fn = get_recursive_character_text_splitter(
    model_serving_endpoint=chunking_config.embedding_model_endpoint,
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

# If you want to implement retrieval strategies such as presenting the entire document vs. the chunk to the LLM, include `doc_content` which contains the doc's full parsed text.  By default this is not included because the size of doc_content can be quite large and cause performance issues.
# propagate_columns = [
#     field.name
#     for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
# ]

# COMMAND ----------

# MAGIC %md
# MAGIC 🚫 Run the chunking function within Spark

# COMMAND ----------

from cookbook_utils.chunk_docs import compute_chunks

chunked_docs_table = compute_chunks(
    # The source documents table.
    docs_table=storage_config.parsed_docs_table,
    # The column containing the documents to be chunked.
    doc_column="doc_content",
    # The chunking function that takes a string (document) and returns a list of strings (chunks).
    chunk_fn=chunk_fn,
    # Choose which columns to propagate from the docs table to chunks table. `doc_uri` column is required we can propagate the original document URL to the Agent's web app.
    propagate_columns=propagate_columns,
    # By default, the chunked_docs_table will be written to `{docs_table}_chunked`.
    chunked_docs_table=storage_config.chunked_docs_table,
    # Pass the Notebook's Spark context
    spark=spark
)

display(spark.read.table(chunked_docs_table))

# Log to MLflow
mlflow.log_input(
    mlflow.data.load_delta(
        table_name=storage_config.chunked_docs_table,
        name=storage_config.chunked_docs_table.replace("`", ""),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline step 3: Create the vector index
# MAGIC
# MAGIC In this step, we'll embed the documents to compute the vector index over the chunks and create our retriever index that will be used to query relevant documents to the user question.  The embedding pipeline is handled within Databricks Vector Search using [Delta Sync](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

# COMMAND ----------

# MAGIC %md
# MAGIC 🚫 Run the embedding pipeline within Vector Search

# COMMAND ----------

from cookbook_utils.build_retriever_index import build_retriever_index

retriever_index_result = build_retriever_index(
    # Spark requires `` to escape names with special chars, VS client does not.
    chunked_docs_table_name=storage_config.chunked_docs_table.replace("`", ""),
    primary_key="chunk_id",
    embedding_source_column="content_chunked",
    vector_search_endpoint=storage_config.vector_search_endpoint,
    vector_search_index_name=storage_config.vector_index,
    # Must match the embedding endpoint you used to chunk your documents
    embedding_endpoint_name=chunking_config.embedding_model_endpoint,
    # Set to true to re-create the vector search endpoint when re-running.  If set to True, syncing will not work if re-run the pipeline and change the schema of chunked_docs_table_name.
    force_delete_index_before_create=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫 End the MLflow run

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

from cookbook_utils import get_table_url
print()
print(f"Parsed docs table: {get_table_url(storage_config.parsed_docs_table)}\n")
print(f"Chunked docs table: {get_table_url(storage_config.chunked_docs_table)}\n")
print(f"Vector search index: {get_table_url(storage_config.vector_index)}\n")
