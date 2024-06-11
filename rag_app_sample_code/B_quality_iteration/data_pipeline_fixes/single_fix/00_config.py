# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny

# COMMAND ----------

# MAGIC %run ./shared_utilities

# COMMAND ----------

# MAGIC %run ../../../00_global_config

# COMMAND ----------

# MAGIC %md ## Configuration for this pipeline
# MAGIC
# MAGIC 1. `CHECKPOINTS_VOLUME_PATH`: Unity Catalog Volume to store Spark streaming checkpoints

# COMMAND ----------

# Temporary location to store Spark streaming checkpoints
# This must a UC Volume; suggest keeping as the default value.
CHECKPOINTS_VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/checkpoints"

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## New Parsing/Chunking/Embedding configuration to try
# MAGIC Modify `configuration` to test a different strategy.  Replace `config_short_name` with a unique name, as it will be prepended to the output tables + vector index.   
# MAGIC
# MAGIC The below cell has a variety of pre-configured strategies that you can try.  Simply uncomment the one you want to test.  You can view more details about each strategy in the Notebooks inside `./supported_configs/`.
# MAGIC
# MAGIC If you want to test multiple configurations at once, see the `../data_pipeline_sweeps/` folder.

# COMMAND ----------

configuration = {
    # Short name to identify this data pipeline by in the chain evaluation
    # `larger_chunk_size` is an example
    "config_short_name": "chunk_size_4096",
    # Initial configuration that matches the POC settings
    # Vector Search index configuration
    "vectorsearch_config": {
        # Pipeline execution mode.
        # TRIGGERED: If the pipeline uses the triggered execution mode, the system stops processing after successfully refreshing the source table in the pipeline once, ensuring the table is updated based on the data available when the update started.
        # CONTINUOUS: If the pipeline uses continuous execution, the pipeline processes new data as it arrives in the source table to keep vector index fresh.
        "pipeline_type": "TRIGGERED",
    },
    # Embedding model to use
    # Tested configurations are available in the `supported_configs/embedding_models` Notebook
    "embedding_config": {
        # Model Serving endpoint name
        "embedding_endpoint_name": "databricks-gte-large-en",
        "embedding_tokenizer": {
            # Name of the embedding model that the tokenizer recognizes
            "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
            # Name of the tokenizer, either `hugging_face` or `tiktoken`
            "tokenizer_source": "hugging_face",
        },
    },
    # Parsing and chunking configuration
    # Databricks provides several default implementations for parsing and chunking documents.  See supported configurations in the `supported_configs/parser_chunker_strategies` Notebook, which are repeated below for ease of use.
    # You can only enable a single `file_format` and `parser` at once
    # You can also add a custom parser/chunker implementation by following the instructions in README.md
    "pipeline_config": {
        # File format of the source documents
        "file_format": "pdf",
        # Parser to use (must be present in `parser_library` Notebook)
        # "parser": {"name": "pypdf", "config": {}},
        "parser": {"name": "pymupdf", "config": {}},
        # "parser": {"name": "pymupdf_markdown", "config": {}},
        # "parser": {
        #     "name": "unstructuredPDF",
        #     "config": {
        #         "strategy": "hi_res",  # optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #         "hi_res_model_name": "yolox",  # optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #         "use_premium_features": False,  # optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #         "api_key": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #         "api_url": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #     },
        # },
        ## DocX
        # File format of the source documents
        # "file_format": "docx",
        # Parser to use (must be present in `parser_library` Notebook)
        # "parser": {"name": "pypandocDocX", "config": {}},
        # "parser": {
        #        "name": "unstructuredDocX",
        #        "config": {
        #            "strategy" : "hi_res",           #optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #            "hi_res_model_name": "yolox",  #optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #            "use_premium_features": False,  #optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #            "api_key": "", #dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #            "api_url": "", #dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #        },
        # },
        ## PPTX
        # File format of the source documents
        # "file_format": "pptx",
        # Parser to use (must be present in `parser_library` Notebook)
        # "parser": {
        #        "name": "unstructuredPPTX",
        #        "config": {
        #            "strategy" : "hi_res",           #optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #            "hi_res_model_name": "yolox",  #optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #            "use_premium_features": False,  #optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #            "api_key": "", # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #            "api_url": "", # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #       },
        # },
        ## HTML
        # "file_format": "html",
        # "parser": {"name": "html_to_markdown", "config": {}},
        # Chunker to use (must be present in `chunker_library` Notebook).  See supported configurations in the `supported_configs/parser_chunker_strategies` Notebook, which are repeated below for ease of use.
        ## JSON
        # "file_format": "json",
        # "parser": {
        #     "name": "json",
        #     "config": {
        #         # The key of the JSON file that contains the content that should be chunked
        #         # All other keys will be passed through
        #         "content_key": "html_content"
        #     },
        # },
        "chunker": {
            ## Split on number of tokens
            "name": "langchain_recursive_char",
            "config": {
                "chunk_size_tokens": 4096,
                "chunk_overlap_tokens": 256,
            },
            ## Split on Markdown headers
            # "name": "langchain_markdown_headers",
            # "config": {
            #     # Include the markdown headers in each chunk?
            #     "include_headers_in_chunks": True,
            # },
            ## Semantic chunk splitter
            # "name": "semantic",
            # "config": {
            #     # Include the markdown headers in each chunk?
            #     "max_chunk_size": 500,
            #     "split_distance_percentile": .95,
            #     "min_sentences": 3
            # },
            "output_table": {
                # The parser function returns a Dict[str, str].  If true, all keys in this dictionary other than 'parsed_content' will be included in the chunk table.  Use this if you extract metadata in your parser that you want to use as a filter in your Vector Search index.
                "include_parser_metadata_as_columns": False,
                # If true, the 'parsed_content' in the Dict[str, str] returned by the parser will be included in the chunk table.  `include_parser_metadata_in_chunk_table` must be True or this option is ignored.
                "include_parent_doc_content_as_column": False,
            },
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the current cofiguration to the POC configuration
# MAGIC
# MAGIC Double check the comparison to make sure that you haven't accidentally changed a setting you didn't intend to.

# COMMAND ----------

import mlflow
# Get the POC's data pipeline configuration
runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_DATA_PIPELINE_RUN_NAME}'", output_format="list")

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_DATA_PIPELINE_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

poc_config = mlflow.artifacts.load_dict(f"{poc_run.info.artifact_uri}/data_pipeline_config.json")

# COMMAND ----------

# Show the differences
differences = compare_dicts(poc_config, configuration)
for item in differences:
    key, old, new = item
    if key != 'config_short_name':
        print(f"Key: {key}")
        print(f"   Changed from: {old}")
        print(f"             to: {new}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the new configuration for later steps to use

# COMMAND ----------

config_short_name = configuration['config_short_name']
mlflow_run_name = f"data_pipeline_{config_short_name}"

# Names of the output Delta Tables tables & Vector Search index
destination_tables_config = {
    # Staging table with the raw files & metadata
    "raw_files_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{config_short_name}_raw_files_bronze",
    # Parsed documents
    "parsed_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{config_short_name}_parsed_docs_silver",
    # Chunked documents that are loaded into the Vector Index
    "chunked_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{config_short_name}_chunked_docs_gold",
    # Destination Vector Index
    "vectorsearch_index_name": f"{UC_CATALOG}.{UC_SCHEMA}.{config_short_name}_chunked_docs_gold_index",
    # Streaming checkpoints, used to only process each file once
    "checkpoint_path": f"{CHECKPOINTS_VOLUME_PATH}/{config_short_name}",
}

vectorsearch_config = configuration['vectorsearch_config']
embedding_config = configuration['embedding_config']
pipeline_config = configuration['pipeline_config']

print(f"Using config: {config_short_name}\n")
print(f"Config settings: {configuration}\n")
print(f"Writing to: {destination_tables_config}")


# COMMAND ----------

print(f"MLflow Run name to use in the evaluation notebook: {mlflow_run_name}")
