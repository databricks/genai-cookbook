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
# MAGIC ### 👉 START HERE: How to Use This Notebook
# MAGIC
# MAGIC Follow these steps to build and refine your data pipeline's quality:
# MAGIC
# MAGIC 1. **Build a v0 index with default settings**
# MAGIC     - Configure the data source and destination tables in the `1️⃣ 📂 Data source & destination configuration` cells
# MAGIC     - Press `Run All` to create the vector index.
# MAGIC
# MAGIC     *Note: While you can adjust the other settings and modify the parsing/chunking code, we suggest doing so only after evaluating your Agent's quality so you can make improvements that specifically address root causes of quality issues.*
# MAGIC
# MAGIC 2. **Use later notebooks to integrate the retriever into an the agent and evaluate the agent/retriever's quality.**
# MAGIC
# MAGIC 3. **If the evaluation results show retrieval issues as a root cause, use this notebook to iterate on your data pipeline's code & config.** Below are some potential fixes you can try, see the AI Cookbook's [debugging retrieval issues](https://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-1-retrieval.html) section for details.**
# MAGIC     - Add missing, but relevant source documents into in the index.
# MAGIC     - Resolve any conflicting information in source documents.
# MAGIC     - Adjust the data pipeline configuration:
# MAGIC       - Modify chunk size or overlap.
# MAGIC       - Experiment with different embedding models.
# MAGIC     - Adjust the data pipeline code:
# MAGIC       - Create a custom parser or use different parsing libraries.
# MAGIC       - Develop a custom chunker or use different chunking techniques.
# MAGIC       - Extract additional metadata for each document.
# MAGIC     - Adjust the Agent's code/config in subsequent notebooks:
# MAGIC       - Change the number of documents retrieved (K).
# MAGIC       - Try a re-ranker.
# MAGIC       - Use hybrid search.
# MAGIC       - Apply extracted metadata as filters.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Important note:** Throughout this notebook, we indicate which cells you:
# MAGIC - ✅✏️ *should* customize - these cells contain code & config with business logic that you should edit to meet your requirements & tune quality
# MAGIC - 🚫✏️ *typically will not* customize - these cells contain boilerplate code required to execute the pipeline
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!  You CAN change these cells, but if this is the first time using this notebook, we suggest not doing so.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Python libraries (Databricks Notebook only)
# MAGIC
# MAGIC 🚫✏️ Only modify if you need additional packages in your code changes to the document parsing or chunking logic.
# MAGIC
# MAGIC Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
# MAGIC Versions of open source packages are locked since package authors often make backwards compatible changes

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC %pip install -qqqq -U -r requirements_datapipeline.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Databricks (Local IDE only)
# MAGIC
# MAGIC If running from an IDE with [`databricks-connect`](https://docs.databricks.com/en/dev-tools/databricks-connect/python/index.html), connect to a Spark session & install the necessary packages on that cluster.

# COMMAND ----------

from cookbook.databricks_utils import get_cluster_url
from cookbook.databricks_utils import get_active_cluster_id
from cookbook.databricks_utils.install_cluster_library import install_requirements

# UNCOMMENT TO INSTALL PACKAGES ON THE ACTIVE CLUSTER; this is code that is not super battle tested.
# cluster_id = get_active_cluster_id()
# print(f"Installing packages on the active cluster: {get_cluster_url(cluster_id)}")


# install_requirements(cluster_id, "requirements.txt")
# install_requirements(cluster_id, "requirements_datapipeline.txt")

# THIS MUST BE DONE MANUALLY! TODO: Automate it.
# - Go to openai_sdk_agent_app_sample_code/
# - Run `poetry build`
# - Copy the wheel file to a UC Volume or Workspace folder
# - Go to the cluster's Libraries page and install the wheel file as a new library

# Get Spark session if using Databricks Connect from an IDE
from mlflow.utils import databricks_utils as du

if not du.is_in_databricks_notebook():
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1️⃣ 📂 Data source & destination configuration

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Configure the data pipeline's source location.
# MAGIC
# MAGIC Choose a [Unity Catalog Volume](https://docs.databricks.com/en/volumes/index.html) containing PDF, HTML, etc documents to be parsed/chunked/embedded.
# MAGIC
# MAGIC - `uc_catalog_name`: Name of the Unity Catalog.
# MAGIC - `uc_schema_name`: Name of the Unity Catalog schema.
# MAGIC - `uc_volume_name`: Name of the Unity Catalog volume.
# MAGIC
# MAGIC Running this cell with validate that the UC Volume exists, trying to create it if not.
# MAGIC

# COMMAND ----------

from cookbook.config.data_pipeline.uc_volume_source import UCVolumeSourceConfig

# Configure the UC Volume that contains the source documents
source_config = UCVolumeSourceConfig(
    # uc_catalog_name="REPLACE_ME", # REPLACE_ME
    # uc_schema_name="REPLACE_ME", # REPLACE_ME
    # uc_volume_name=f"REPLACE_ME", # REPLACE_ME
    uc_catalog_name="casaman_ssa", # REPLACE_ME
    uc_schema_name="demos", # REPLACE_ME
    uc_volume_name="volume_databricks_documentation", # REPLACE_ME
)

# Check if volume exists, create otherwise
is_valid, msg = source_config.create_or_validate_volume()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Configure the data pipeline's output location.
# MAGIC  
# MAGIC Choose where the data pipeline outputs the parsed, chunked, and embedded documents.
# MAGIC
# MAGIC Required parameters:
# MAGIC * `uc_catalog_name`: Unity Catalog name where tables will be created
# MAGIC * `uc_schema_name`: Schema name within the catalog 
# MAGIC * `base_table_name`: Core name used as prefix for all generated tables
# MAGIC * `vector_search_endpoint`: Vector Search endpoint to store the index
# MAGIC
# MAGIC Optional parameters:
# MAGIC * `docs_table_postfix`: Suffix for the parsed documents table (default: "docs")
# MAGIC * `chunked_table_postfix`: Suffix for the chunked documents table (default: "docs_chunked") 
# MAGIC * `vector_index_postfix`: Suffix for the vector index (default: "docs_chunked_index")
# MAGIC * `version_suffix`: Version identifier (e.g. 'v1', 'test') to maintain multiple versions
# MAGIC
# MAGIC The generated tables follow this naming convention:
# MAGIC * Parsed docs: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{docs_table_postfix}__{version_suffix}
# MAGIC * Chunked docs: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{chunked_table_postfix}__{version_suffix}
# MAGIC * Vector index: {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{vector_index_postfix}__{version_suffix}
# MAGIC
# MAGIC *Note: If you are comparing different chunking/parsing/embedding strategies, set the `version_suffix` parameter to maintain multiple versions of the pipeline output with the same base_table_name.*
# MAGIC
# MAGIC *Databricks suggests sharing a Vector Search endpoint across multiple agents.*

# COMMAND ----------

from cookbook.config.data_pipeline.data_pipeline_output import DataPipelineOuputConfig

# Output configuration
output_config = DataPipelineOuputConfig(
    # Required parameters
    uc_catalog_name=source_config.uc_catalog_name, # usually same as source volume catalog, by default is the same as the source volume catalog
    uc_schema_name=source_config.uc_schema_name, # usually same as source volume schema, by default is the same as the source volume schema
    #base_table_name=source_config.uc_volume_name, # usually similar / same as the source volume name; by default, is the same as the volume_name
    base_table_name="test_product_docs", # usually similar / same as the source volume name; by default, is the same as the volume_name
    # vector_search_endpoint="REPLACE_ME", # Vector Search endpoint to store the index
    vector_search_endpoint="one-env-shared-endpoint-3", # Vector Search endpoint to store the index

    # Optional parameters, showing defaults
    docs_table_postfix="docs",              # default value is `docs`
    chunked_table_postfix="docs_chunked",   # default value is `docs_chunked`
    vector_index_postfix="docs_chunked_index", # default value is `docs_chunked_index`
    version_suffix="v2"                     # default is None

    # Output tables / indexes follow this naming convention:
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{docs_table_postfix}__{version_suffix}
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{chunked_table_postfix}__{version_suffix}
    # {uc_catalog_name}.{uc_schema_name}.{base_table_name}_{vector_index_postfix}__{version_suffix}
)

# Alternatively, you can directly pass in the UC locations of the tables / indexes
# output_config = DataPipelineOuputConfig(
#     chunked_docs_table="catalog.schema.docs_chunked",
#     parsed_docs_table="catalog.schema.parsed_docs",
#     vector_index="catalog.schema.docs_chunked_index",
#     vector_search_endpoint="REPLACE_ME",
# )

# Check UC locations exist
is_valid, msg = output_config.validate_catalog_and_schema()
if not is_valid:
    raise Exception(msg)

# Check Vector Search endpoint exists
is_valid, msg = output_config.validate_vector_search_endpoint()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Configure chunk size and the embedding model.
# MAGIC
# MAGIC **Chunk size and overlap** control how a larger document is turned into smaller chunks that can be processed by an embedding model.  See the AI Cookbook [chunking deep dive](https://ai-cookbook.io/nbs/3-deep-dive-data-pipeline.html#chunking) for more details.
# MAGIC
# MAGIC **The embedding model** is an AI model that is used to identify the most similar documents to a given user's query.  See the AI Cookbook [embedding model deep dive](https://ai-cookbook.io/nbs/3-deep-dive-data-pipeline.html#embedding-model) for more details.
# MAGIC
# MAGIC This notebook supports the following [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/embeddings`/.  If you want to try another model, you will need to modify the `utils/get_recursive_character_text_splitter` Notebook to add support.
# MAGIC - `databricks-gte-large-en` or `databricks-bge-large-en`
# MAGIC - Azure OpenAI or OpenAI External Model of type `text-embedding-ada-002`, `text-embedding-3-small` or `text-embedding-3-large`

# COMMAND ----------

from cookbook.config.data_pipeline.recursive_text_splitter import RecursiveTextSplitterChunkingConfig

chunking_config = RecursiveTextSplitterChunkingConfig(
    embedding_model_endpoint="databricks-gte-large-en",  # A Model Serving endpoint supporting the /llm/v1/embeddings task
    chunk_size_tokens=1024,
    chunk_overlap_tokens=256,
)

# Validate the embedding endpoint & chunking config
is_valid, msg = chunking_config.validate_embedding_endpoint()
if not is_valid:
    raise Exception(msg)

is_valid, msg = chunking_config.validate_chunk_size_and_overlap()
if not is_valid:
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ Write the data pipeline configuration to a YAML
# MAGIC
# MAGIC This allows the configuration to be loaded referenced by the Agent's notebook.

# COMMAND ----------

from cookbook.config.data_pipeline import DataPipelineConfig
from cookbook.config import serializable_config_to_yaml_file

data_pipeline_config = DataPipelineConfig(
    source=source_config,
    output=output_config,
    chunking_config=chunking_config,
)

serializable_config_to_yaml_file(data_pipeline_config, "./configs/data_pipeline_config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🛑 If you are running your initial data pipeline, you do not need to configure anything else, you can just `Run All` the notebook cells before.  You can modify these cells later to tune the quality of your data pipeline by changing the parsing logic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ ⌨️ Data pipeline code
# MAGIC
# MAGIC The code below executes the data pipeline.  You can modify the below code as indicated to implement different parsing or chunking strategies or to extract additional metadata fields

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
# MAGIC ##### ✅✏️ Customize the parsing function
# MAGIC
# MAGIC This default implementation parses PDF, HTML, and DOCX files using open source libraries.  Adjust `file_parser(...)` and `ParserReturnValue` in `cookbook/data_pipeline/default_parser.py` to add change the parsing logic, add support for more file types, or extract additional metadata about each document.

# COMMAND ----------

from cookbook.data_pipeline.default_parser import file_parser, ParserReturnValue

# Print the code of file_parser function for inspection
import inspect
print(inspect.getsource(ParserReturnValue))
print(inspect.getsource(file_parser))


# COMMAND ----------

# MAGIC %md
# MAGIC The below cell is debugging code to test your parsing function on a single record. 

# COMMAND ----------

from cookbook.data_pipeline.parse_docs import load_files_to_df
from pyspark.sql import functions as F


raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_config.volume_path,
)

print(f"Loaded {raw_files_df.count()} files from {source_config.volume_path}.  Files: {source_config.list_files()}")

test_records_dict = raw_files_df.toPandas().to_dict(orient="records")

for record in test_records_dict:
  print()
  print("Testing parsing for file: ", record["path"])
  print()
  test_result = file_parser(raw_doc_contents_bytes=record['content'], doc_path=record['path'], modification_time=record['modificationTime'], doc_bytes_length=record['length'])
  print(test_result)
  break # pause after 1 file.  if you want to test more files, remove the break statement


# COMMAND ----------

# MAGIC %md
# MAGIC 🚫✏️ The below cell is boilerplate code to apply the parsing function using Spark.

# COMMAND ----------

from cookbook.data_pipeline.parse_docs import (
    load_files_to_df,
    apply_parsing_fn,
    check_parsed_df_for_errors,
    check_parsed_df_for_empty_parsed_files
)
from cookbook.data_pipeline.utils.typed_dicts_to_spark_schema import typed_dicts_to_spark_schema
from cookbook.databricks_utils import get_table_url

# Tune this parameter to optimize performance.  More partitions will improve performance, but may cause out of memory errors if your cluster is too small.
NUM_PARTITIONS = 50

# Load the UC Volume files into a Spark DataFrame
raw_files_df = load_files_to_df(
    spark=spark,
    source_path=source_config.volume_path,
).repartition(NUM_PARTITIONS)

# Apply the parsing UDF to the Spark DataFrame
parsed_files_df = apply_parsing_fn(
    raw_files_df=raw_files_df,
    # Modify this function to change the parser, extract additional metadata, etc
    parse_file_fn=file_parser,
    # The schema of the resulting Delta Table will follow the schema defined in ParserReturnValue
    parsed_df_schema=typed_dicts_to_spark_schema(ParserReturnValue),
)

# Write to a Delta Table
parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    output_config.parsed_docs_table
)

# Get resulting table
parsed_files_df = spark.table(output_config.parsed_docs_table)
parsed_files_no_errors_df = parsed_files_df.filter(
    parsed_files_df.parser_status == "SUCCESS"
)

# Show successfully parsed documents
print(f"Parsed {parsed_files_df.count()} / {parsed_files_no_errors_df.count()} documents successfully.  Inspect `parsed_files_no_errors_df` or visit {get_table_url(output_config.parsed_docs_table)} to see all parsed documents, including any errors.")
display(parsed_files_no_errors_df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC Show any parsing failures or successfully parsed files that resulted in an empty document.

# COMMAND ----------


# Any documents that failed to parse
is_error, msg, failed_docs_df = check_parsed_df_for_errors(parsed_files_df)
if is_error:
    display(failed_docs_df.toPandas())
    raise Exception(msg)
    
# Any documents that returned empty parsing results
is_error, msg, empty_docs_df = check_parsed_df_for_empty_parsed_files(parsed_files_df)
if is_error:
    display(empty_docs_df.toPandas())
    raise Exception(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline step 2: Compute chunks of documents
# MAGIC
# MAGIC In this step, we will split our documents into smaller chunks so they can be indexed in our vector database.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### ✅✏️ Chunking logic.
# MAGIC
# MAGIC We provide a default implementation of a recursive text splitter.  To create your own chunking logic, adapt the `get_recursive_character_text_splitter()` function inside `cookbook.data_pipeline.recursive_character_text_splitter.py`.

# COMMAND ----------

from cookbook.data_pipeline.recursive_character_text_splitter import (
    get_recursive_character_text_splitter,
)

# Get the chunking function
recursive_character_text_splitter_fn = get_recursive_character_text_splitter(
    model_serving_endpoint=chunking_config.embedding_model_endpoint,
    chunk_size_tokens=chunking_config.chunk_size_tokens,
    chunk_overlap_tokens=chunking_config.chunk_overlap_tokens,
)

# Determine which columns to propagate from the docs table to the chunks table.

# Get the columns from the parser except for the content
# You can modify this to adjust which fields are propagated from the docs table to the chunks table.
propagate_columns = [
    field.name
    for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
    if field.name != "content"
]

# If you want to implement retrieval strategies such as presenting the entire document vs. the chunk to the LLM, include `contentich contains the doc's full parsed text.  By default this is not included because the size of contcontentquite large and cause performance issues.
# propagate_columns = [
#     field.name
#     for field in typed_dicts_to_spark_schema(ParserReturnValue).fields
# ]

# COMMAND ----------

# MAGIC %md
# MAGIC 🚫✏️ Run the chunking function within Spark

# COMMAND ----------

from cookbook.data_pipeline.chunk_docs import apply_chunking_fn
from cookbook.databricks_utils import get_table_url

# Tune this parameter to optimize performance.  More partitions will improve performance, but may cause out of memory errors if your cluster is too small.
NUM_PARTITIONS = 50

# Load parsed docs
parsed_files_df = spark.table(output_config.parsed_docs_table).repartition(NUM_PARTITIONS)

chunked_docs_df = chunked_docs_table = apply_chunking_fn(
    # The source documents table.
    parsed_docs_df=parsed_files_df,
    # The chunking function that takes a string (document) and returns a list of strings (chunks).
    chunking_fn=recursive_character_text_splitter_fn,
    # Choose which columns to propagate from the docs table to chunks table. `doc_uri` column is required we can propagate the original document URL to the Agent's web app.
    propagate_columns=propagate_columns,
)

# Write to Delta Table
chunked_docs_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(output_config.chunked_docs_table)

# Get resulting table
chunked_docs_df = spark.table(output_config.chunked_docs_table)

# Show number of chunks created
print(f"Created {chunked_docs_df.count()} chunks.  Inspect `chunked_docs_df` or visit {get_table_url(output_config.chunked_docs_table)} to see the results.")

# enable CDC feed for VS index sync
cdc_results = spark.sql(f"ALTER TABLE {output_config.chunked_docs_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# Show chunks
display(chunked_docs_df.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ Pipeline step 3: Create the vector index
# MAGIC
# MAGIC In this step, we'll embed the documents to compute the vector index over the chunks and create our retriever index that will be used to query relevant documents to the user question.  The embedding pipeline is handled within Databricks Vector Search using [Delta Sync](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

# COMMAND ----------

from cookbook.data_pipeline.build_retriever_index import build_retriever_index
from cookbook.databricks_utils import get_table_url

is_error, msg = retriever_index_result = build_retriever_index(
    # Spark requires `` to escape names with special chars, VS client does not.
    chunked_docs_table_name=output_config.chunked_docs_table.replace("`", ""),
    vector_search_endpoint=output_config.vector_search_endpoint,
    vector_search_index_name=output_config.vector_index,

    # Must match the embedding endpoint you used to chunk your documents
    embedding_endpoint_name=chunking_config.embedding_model_endpoint,

    # Set to true to re-create the vector search endpoint when re-running the data pipeline.  If set to True, syncing will not work if re-run the pipeline and change the schema of chunked_docs_table_name.  Keeping this as False will allow Vector Search to avoid recomputing embeddings for any row with that has a chunk_id that was previously computed.
    force_delete_index_before_create=False,
)
if is_error:
    raise Exception(msg)
else:
    print("NOTE: This cell will complete before the vector index has finished syncing/embedding your chunks & is ready for queries!")
    print(f"View sync status here: {get_table_url(output_config.vector_index)}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ Print links to view the resulting tables/index

# COMMAND ----------

from cookbook.databricks_utils import get_table_url

print()
print(f"Parsed docs table: {get_table_url(output_config.parsed_docs_table)}\n")
print(f"Chunked docs table: {get_table_url(output_config.chunked_docs_table)}\n")
print(f"Vector search index: {get_table_url(output_config.vector_index)}\n")