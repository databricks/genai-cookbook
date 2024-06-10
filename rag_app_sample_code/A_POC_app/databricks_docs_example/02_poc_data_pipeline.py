# Databricks notebook source
# MAGIC %md
# MAGIC # POC JSON file Data Preparation Pipeline
# MAGIC
# MAGIC This is an example notebook that provides a **starting point** for building a POC data pipeline which uses the configuration from `00_config`: 
# MAGIC - Loads and parses PDF files from a UC Volume.
# MAGIC - Chunks the data.
# MAGIC - Converts the chunks into embeddings.
# MAGIC - Stores the embeddings in a Databricks Vector Search Index. 
# MAGIC
# MAGIC After you have initial feedback from your stakeholders, you can easily adapt and tweak this pipeline to fit more advanced logic. For example, if your retrieval quality is low, you could experiment with different parsing and chunk sizes once you gain working knowledge of your data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -qqqq -U databricks-vectorsearch transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0 mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from typing import TypedDict, Dict
import warnings
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

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

mlflow.end_run()
# Start MLflow logging
run = mlflow.start_run(run_name=POC_DATA_PIPELINE_RUN_NAME)

# Tag the run
mlflow.set_tag("type", "data_pipeline")

# Set the parameters
mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))
mlflow.log_params(_flatten_nested_params({"destination_tables": destination_tables_config}))

# Log the configs as artifacts for later use
mlflow.log_dict(destination_tables_config, "destination_tables_config.json")
mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load the files from the UC Volume
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Bronze table.
# MAGIC
# MAGIC **NOTE:** You will have to upload some PDF files to this volume. See the `sample_pdfs` folder of this repo for some example PDFs to upload to the UC Volume.
# MAGIC
# MAGIC TODO: Another notebook to load sample PDFs if the customer does't have them

# COMMAND ----------

# Load the raw riles
raw_files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", f"*.{pipeline_config.get('file_format')}")
    .load(SOURCE_PATH)
)

# Save to a table
raw_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    destination_tables_config["raw_files_table_name"]
)

# reload to get correct lineage in UC
raw_files_df = spark.read.table(destination_tables_config["raw_files_table_name"])

# For debugging, show the list of files, but hide the binary content
display(raw_files_df.drop("content"))

# Check that files were present and loaded
if raw_files_df.count() == 0:
    display(
        f"`{SOURCE_PATH}` does not contain any files.  Open the volume and upload at least file."
    )
    raise Exception(f"`{SOURCE_PATH}` does not contain any files.")

tag_delta_table(destination_tables_config["raw_files_table_name"], data_pipeline_config)

mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("raw_files_table_name")), context="raw_files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Parse the PDF files into text
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Silver table.
# MAGIC
# MAGIC Although not reccomended for your POC, if you want to change the parsing library or adjust it's settings, modify the contents of the `parse_bytes_pypdf` UDF.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # JSON files
# MAGIC
# MAGIC `content_key`: JSON key containing the string content that should be chunked
# MAGIC
# MAGIC All other keys will be passed through as-is
# MAGIC

# COMMAND ----------

import json

class ParserReturnValue(TypedDict):
    doc_parsed_contents: Dict[str, str]
    parser_status: str


def parse_bytes_json(
    raw_doc_contents_bytes: bytes,
    content_key: str
) -> ParserReturnValue:

    try:
        # Decode the raw bytes from Spark
        json_str = raw_doc_contents_bytes.decode("utf-8")
        # Load the JSON contained in the bytes
        json_data = json.loads(json_str)
        
        # Load the known key to `parsed_content`
        json_data['parsed_content'] = json_data[content_key]
        
        # Remove that key
        del json_data[content_key]
        
        output = json_data
        status = "SUCCESS"

    except json.JSONDecodeError as e:
        status = f"JSON decoding failed: {e}"
        output = {
            "parsed_content": "",
        }
        warnings.warn(status)
    except UnicodeDecodeError as e:
        status = f"Unicode decoding failed: {e}"
        output = {
            "parsed_content": "",
        }
        warnings.warn(status)
    except Exception as e:
        status = f"An unexpected error occurred: {e}"
        output = {
            "parsed_content": "",
        }

    return {
        "doc_parsed_contents": output,
        "parser_status": status,
    }



# COMMAND ----------

# MAGIC %md
# MAGIC ### Parser UDF
# MAGIC
# MAGIC This UDF wraps your parser into a UDF so Spark can parallelize the data processing.

# COMMAND ----------

parser_config = pipeline_config.get("parser")
parser_udf = func.udf(
    partial(
        parse_bytes_json,
        content_key=parser_config.get("config").get("content_key"),
    ),
    returnType=StructType(
        [
            StructField(
                "doc_parsed_contents",
                MapType(StringType(), StringType()),
                nullable=True,
            ),
            StructField("parser_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the parsers in Spark
# MAGIC
# MAGIC This cell runs the configured parsers in parallel via Spark.  Inspect the outputs to verify that parsing is working correctly.

# COMMAND ----------

# Run the parsing
parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content")).drop("content")


# Check and warn on any errors
errors_df = parsed_files_staging_df.filter(
    func.col(f"parsing.parser_status")
    != "SUCCESS"
)

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} documents had parse errors.  Please review.")
    display(errors_df)

# Filter for successfully parsed files
parsed_files_df = parsed_files_staging_df.filter(parsed_files_staging_df.parsing.parser_status == "SUCCESS").withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents")).drop("parsing")

# Write to Delta Table
parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(destination_tables_config["parsed_docs_table_name"])

# reload to get correct lineage in UC
parsed_files_df = spark.table(destination_tables_config["parsed_docs_table_name"])

# Display for debugging
print(f"Parsed {parsed_files_df.count()} documents.")

display(parsed_files_df)

tag_delta_table(destination_tables_config["parsed_docs_table_name"], data_pipeline_config)
mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("parsed_docs_table_name")), context="parsed_docs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Chunk the parsed text
# MAGIC
# MAGIC
# MAGIC In Bronze/Silver/Gold terminology, this is your Gold table.
# MAGIC
# MAGIC Although not reccomended for your POC, if you want to change the chunking library or adjust it's settings, modify the contents of the `parse_bytes_pypdf` UDF.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recursive Token Based Text Splitter
# MAGIC Uses the embedding model's tokenizer to split the document into chunks.
# MAGIC
# MAGIC Per LangChain's docs: This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
# MAGIC
# MAGIC Configuration parameters:
# MAGIC - `chunk_size_tokens`: Number of tokens to include in each chunk
# MAGIC - `chunk_overlap_tokens`: Number of tokens to overlap between chunks e.g., the last `chunk_overlap_tokens` tokens of chunk N are the same as the first `chunk_overlap_tokens` tokens of chunk N+1
# MAGIC
# MAGIC Docs: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# MAGIC
# MAGIC IMPORTANT: You need to ensure that `chunk_size_tokens` + `chunk_overlap_tokens` is LESS THAN your embedding model's context window.

# COMMAND ----------

class ChunkerReturnValue(TypedDict):
    chunked_text: str
    chunker_status: str

def chunk_parsed_content_langrecchar(
    doc_parsed_contents: str, chunk_size: int, chunk_overlap: int, embedding_config
) -> ChunkerReturnValue:
    try:
        # Select the correct tokenizer based on the embedding model configuration
        if (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "hugging_face"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "tiktoken"
        ):
            tokenizer = tiktoken.encoding_for_model(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

        chunks = text_splitter.split_text(doc_parsed_contents)
        return {
            "chunked_text": [doc for doc in chunks],
            "chunker_status": "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "chunked_text": [],
            "chunker_status": f"ERROR: {e}",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunker UDF
# MAGIC
# MAGIC This UDF wraps your chunker into a UDF so Spark can parallelize the data processing.

# COMMAND ----------

chunker_conf = pipeline_config.get("chunker")

chunker_udf = func.udf(
    partial(
        chunk_parsed_content_langrecchar,
        chunk_size=chunker_conf.get("config").get("chunk_size_tokens"),
        chunk_overlap=chunker_conf.get("config").get("chunk_overlap_tokens"),
        embedding_config=embedding_config,
    ),
    returnType=StructType(
        [
            StructField("chunked_text", ArrayType(StringType()), nullable=True),
            StructField("chunker_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the chunker in Spark
# MAGIC
# MAGIC This cell runs the configured chunker in parallel via Spark.  Inspect the outputs to verify that chunking is working correctly.

# COMMAND ----------

# Run the chunker
chunked_files_df = parsed_files_df.withColumn(
    "chunked",
    chunker_udf("doc_parsed_contents.parsed_content"),
)

# Check and warn on any errors
errors_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status != "SUCCESS")

num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} chunks had parse errors.  Please review.")
    display(errors_df)

# Filter for successful chunks
chunked_files_df = chunked_files_df.filter(chunked_files_df.chunked.chunker_status == "SUCCESS").select(
    "path",
    func.explode("chunked.chunked_text").alias("chunked_text"),
    func.md5(func.col("chunked_text")).alias("chunk_id")
)

# Append the JSON's metadata to the chunked table

keys_df = spark.read.table(
destination_tables_config.get("parsed_docs_table_name")
)

df_keys_array = keys_df.select(func.explode(func.map_keys(keys_df.doc_parsed_contents)).alias("key")).distinct().toPandas()["key"].tolist()

metadata_df = parsed_files_df

if 'parsed_content' in df_keys_array:
    df_keys_array.remove('parsed_content')

for key in df_keys_array:
    metadata_df = metadata_df.withColumn(key, metadata_df.doc_parsed_contents[key])

metadata_df = metadata_df.drop("doc_parsed_contents").withColumnRenamed("path", "path_2")

chunked_files_df = chunked_files_df.join(metadata_df, func.expr("path_2 = path"), "inner").drop("path_2")

# Write to Delta Table
chunked_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    destination_tables_config["chunked_docs_table_name"]
)


# Enable CDC for Vector Search Delta Sync
spark.sql(
    f"ALTER TABLE {destination_tables_config['chunked_docs_table_name']} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(f"Produced a total of {chunked_files_df.count()} chunks.")

# Display without the parent document text - this is saved to the Delta Table
display(chunked_files_df)

tag_delta_table(destination_tables_config["chunked_docs_table_name"], data_pipeline_config)

mlflow.log_input(mlflow.data.load_delta(table_name=destination_tables_config.get("chunked_docs_table_name")), context="chunked_docs")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ## Embed documents & sync to Vector Search index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Get the vector search index
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# DBTITLE 1,Index Management Workflow
force_delete = False

def find_index(endpoint_name, index_name):
    all_indexes = vsc.list_indexes(name=VECTOR_SEARCH_ENDPOINT).get("vector_indexes", [])
    return destination_tables_config["vectorsearch_index_name"] in map(lambda i: i.get("name"), all_indexes)

if find_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=destination_tables_config["vectorsearch_index_name"]):
    if force_delete:
        vsc.delete_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=destination_tables_config["vectorsearch_index_name"])
        create_index = True
    else:
        create_index = False
else:
    create_index = True

if create_index:
    print("Embedding docs & creating Vector Search Index, this can take 15 minutes or much longer if you have a larger number of documents.")
    print(f'Check status at: {get_table_url(destination_tables_config["vectorsearch_index_name"])}')

    vsc.create_delta_sync_index_and_wait(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=destination_tables_config["vectorsearch_index_name"],
        primary_key="chunk_id",
        source_table_name=destination_tables_config["chunked_docs_table_name"],
        pipeline_type=vectorsearch_config['pipeline_type'],
        embedding_source_column="chunked_text",
        embedding_model_endpoint_name=embedding_config['embedding_endpoint_name']
    )

tag_delta_table(destination_tables_config["vectorsearch_index_name"], data_pipeline_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View index status & output tables
# MAGIC
# MAGIC You can view the index status and how to query the index at the URL below.

# COMMAND ----------

index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=destination_tables_config["vectorsearch_index_name"])

print(f'Vector index: {get_table_url(destination_tables_config["vectorsearch_index_name"])}')
print("\nOutput tables:\n")
print(f"Bronze Delta Table w/ raw files: {get_table_url(destination_tables_config['raw_files_table_name'])}")
print(f"Silver Delta Table w/ parsed files: {get_table_url(destination_tables_config['parsed_docs_table_name'])}")
print(f"Gold Delta Table w/ chunked files: {get_table_url(destination_tables_config['chunked_docs_table_name'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the index

# COMMAND ----------

# DBTITLE 1,Testing the Index
index.similarity_search(columns=["chunked_text", "chunk_id", "path"], query_text="your query text")

# COMMAND ----------

chain_config = {
    "databricks_resources": {
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
    },
    "retriever_config": {
        "vector_search_index": destination_tables_config[
            "vectorsearch_index_name"
        ],
        "data_pipeline_tag": "poc",
    }
}

mlflow.log_dict(chain_config, "chain_config.json")

mlflow.end_run()
