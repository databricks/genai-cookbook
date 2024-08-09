# Databricks notebook source
# MAGIC %md
# MAGIC # POC PPTX Data Preparation Pipeline
# MAGIC
# MAGIC This is a POC data preperation that provides uses the configuration from `00_config` to build a data pipeline that loads, parses, chunks, and embeds PPTX files from a UC Volume into a Databricks Vector Search Index.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -qqqq -U markdownify==0.12.1 "unstructured[local-inference, all-docs]==0.14.4" unstructured-client==0.22.0 pdfminer==20191125 nltk==3.8.1 databricks-vectorsearch transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.2 mlflow mlflow-skinny

# COMMAND ----------

from typing import List
def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(
        1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    )

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e



# COMMAND ----------

install_apt_get_packages(["poppler-utils", "tesseract-ocr"])

# COMMAND ----------

dbutils.library.restartPython()

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

# MAGIC %md TODO: verify that the catalog & vector index exist
# MAGIC
# MAGIC consider creating a vector search endpoint with their name
# MAGIC
# MAGIC Your user name, used to ensure that multiple users who run this demo don't overwrite each other
# MAGIC CURRENT_USERNAME = spark.sql("SELECT current_user()").collect()[0][0].split("@")[0].replace(".", "_")

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
# MAGIC """ The Unstructured PPTX parser, which enables both free local parsing as well as premium API-based parsing of .pptx files.
# MAGIC
# MAGIC     Parameters:
# MAGIC     - strategy (str): The strategy to use for parsing the PPTX document. Options include:
# MAGIC         - "ocr_only": Runs the document through Tesseract for OCR and then processes the raw text. Recommended for documents with multiple columns that do not have extractable text. Falls back to "fast" if Tesseract is not available and the document has extractable text.
# MAGIC         - "fast": Extracts text and processes the raw text. Recommended for most cases where the PPTX has extractable text. Falls back to "ocr_only" if the text is not extractable.
# MAGIC         - "hi_res": Identifies the layout of the document using a specified model (e.g., detectron2_onnx). Uses the document layout to gain additional information about document elements. Recommended if your use case is highly sensitive to correct classifications for document elements. Falls back to "ocr_only" if the specified model is not available.
# MAGIC         The default strategy is "hi_res".
# MAGIC     - hi_res_model_name (str): The name of the model to use for the "hi_res" strategy. Options include:
# MAGIC         - "detectron2_onnx": A Computer Vision model by Facebook AI that provides object detection and segmentation algorithms with ONNX Runtime. It is the fastest model for the "hi_res" strategy.
# MAGIC         - "yolox": A single-stage real-time object detector that modifies YOLOv3 with a DarkNet53 backbone.
# MAGIC         - "yolox_quantized": Runs faster than YoloX and its speed is closer to Detectron2.
# MAGIC         The default model is "yolox".
# MAGIC     - use_premium_features (bool): Whether to use premium, proprietary models and features for document parsing. These models may offer better accuracy or additional features compared to open-source models, but require an API key and endpoint URL for access. Set to `True` to enable the use of premium models. The default is `False`.
# MAGIC         The default setting is False.
# MAGIC     - api_key (str): The API key required to access the premium parsing engine. This is only needed if `use_premium_models` is set to `True`. This key authenticates the requests to the premium API service.
# MAGIC         The default setting is "".
# MAGIC     - api_url (str): The URL of the API endpoint for accessing premium parsing engine. This should be provided if `use_premium_models` is set to `True`. For Unstructured-hosted SaaS API, the format should be https://{{UNSTRUCT_SAAS_API_TENANT_ID}}.api.unstructuredapp.io/ and https://api.unstructured.io/general/v0/general/ for the Unstructured-hosted, capped-usage, free API.
# MAGIC         The default setting is "".
# MAGIC     """    

# COMMAND ----------

class ParserReturnValue(TypedDict):
    doc_parsed_contents: Dict[str, str]
    parser_status: str


def parse_bytes_unstructuredPPTX(
    raw_doc_contents_bytes: bytes,
    strategy:str = "hi_res",       #Strategy to use for parsing. Options: "hi_res", "ocr_only", "fast"
    hi_res_model_name:str="yolox", #hi_res model name. Options  "yolox", "yolox_quantized", "detectron2_onnx"
    doc_name: str = "filename_unavailable.pptx", # TODO: can the "doc_uri" param be passed into the parse_bytes method?
    use_premium_features: bool|None = None,     # optional; allows user to toggle/on premium features on a per call basis
    api_key:str="", #optional; needed for premium features
    api_url:str="",  #optional; needed for premium features
    **kwargs
) -> ParserReturnValue:
    from unstructured.partition.pptx import partition_pptx
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared
    from unstructured_client.models.errors import SDKError
    from unstructured.staging.base import elements_from_json
    import json        
    import io
    from markdownify import markdownify as md
    
    try:
        reconstructed_file = io.BytesIO(raw_doc_contents_bytes)
        #If args are None, then set them to default values
        if (strategy is None): strategy = "hi_res" 
        if (hi_res_model_name is None ): hi_res_model_name = "yolox"
        if (doc_name is None ): doc_name = "filename_unavailable.pptx"

        parsing_base_config = {
            "strategy": strategy, # mandatory to use ``hi_res`` strategy
            "hi_res_model_name": hi_res_model_name,
            "skip_infer_table_types": [], # file types to skip
            "extract_image_block_types": ["Image", "Table"], # optional
        }
        api_config = {
            **parsing_base_config,
            "files": shared.Files(
                content=raw_doc_contents_bytes,
                file_name=doc_name
            ),
            }

        local_config = {
            **parsing_base_config,
            "file": reconstructed_file,
            "source_format": "docx",
            "infer_table_structure": True,
            "extract_image_block_to_payload": True, # optional
            }
        # The use_premium_features flag routes requests to an Unstructured-hosted or client-hosted Marketplace API (Free, SaaS, or Marketplace VPC)
        
        if use_premium_features:
            try: 
                client = UnstructuredClient(
                    server_url=api_url,
                    api_key_auth=api_key,
                )
                req = shared.PartitionParameters(**api_config)
                resp = client.general.partition(req)
                document_sections = elements_from_json(text=json.dumps(resp.elements))
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the premium Unstructured API: {e}")
        else:
            try: 
                document_sections = partition_pptx(**local_config, **kwargs)
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the Unstructured open source library: {e}")
        text_content = ""
        for section in document_sections:
            # Tables are parsed seperately, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text
        
        output = {
            "parsed_content": text_content,
        }
        value_to_return = output

        return {
            "doc_parsed_contents": output,
            "parser_status": "SUCCESS",
        }
    #TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "doc_parsed_contents": [{"page_number": None, "parsed_content": None}],
            "parser_status": f"ERROR: {e}",
        }
    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parser UDF
# MAGIC
# MAGIC This UDF wraps your parser into a UDF so Spark can parallelize the data processing.

# COMMAND ----------

from functools import partial


parser_config = pipeline_config.get("parser")


parser_udf = func.udf(
    partial(
        parse_bytes_unstructuredPPTX,
        strategy = parser_config.get("config").get("strategy"),
        hi_res_model_name = parser_config.get("config").get("hi_res_model_name"),
        use_premium_features = parser_config.get("config").get("use_premium_features"),
        api_url = parser_config.get("config").get("api_url"),
        api_key = parser_config.get("config").get("api_key"),
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
    return index_name in map(lambda i: i.get("name"), all_indexes)

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
        source_table_name=destination_tables_config["chunked_docs_table_name"].replace("`", ""),
        pipeline_type=vectorsearch_config['pipeline_type'],
        embedding_source_column="chunked_text",
        embedding_model_endpoint_name=embedding_config['embedding_endpoint_name']
    )

tag_delta_table(destination_tables_config["vectorsearch_index_table_name"], data_pipeline_config)

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
