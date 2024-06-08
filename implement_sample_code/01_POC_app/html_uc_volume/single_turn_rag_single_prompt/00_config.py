# Databricks notebook source
# MAGIC %pip install -U -qqqq install pyyaml

# COMMAND ----------

# By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
current_user = spark.sql("SELECT current_user() as username").collect()[0].username

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Configure the following:
# MAGIC 1. `RAG_APP_NAME`: The name of the RAG application.  This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes
# MAGIC 2. `UC_CATALOG` & `UC_SCHEMA`: Unity Catalog Schema where the Delta Tables with the parsed/chunked documents are stored
# MAGIC 3. `UC_MODEL_NAME`: UC location to log/store the chain's model
# MAGIC 4. `VECTOR_SEARCH_ENDPOINT`: Vector Search Endpoint to host the resulting vector index
# MAGIC 5. `SOURCE_PATH`: Unity Catalog Volume with the source documents
# MAGIC
# MAGIC After finalizing your configuration, optionally run `00_validate_config` to check that all locations exist. 

# COMMAND ----------

# The name of the RAG application.  This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes
RAG_APP_NAME = 'my_single_turn_html_rag_app'

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'{current_user.split("@")[0].replace(".", "")}_catalog'
UC_SCHEMA = f'rag_{current_user.split("@")[0].split(".")[0]}'

## UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{current_user.split("@")[0].replace(".", "")}_vector_search'

# Source location for documents
SOURCE_PATH = "/Volumes/rag/testing/raw_data"


# COMMAND ----------

print(f"POC app using the UC catalog/schema {UC_CATALOG}.{UC_SCHEMA} with source data from {SOURCE_PATH} synced to the Vector Search endpoint {VECTOR_SEARCH_ENDPOINT}.  Chain model will be logged to UC as {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # POC Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC ### Config
# MAGIC
# MAGIC First, we configure the data pipeline's default settings.

# COMMAND ----------

data_pipeline_config = {
    # Vector Search index configuration
    "vectorsearch_config": {
        # Pipeline execution mode.
        # TRIGGERED: If the pipeline uses the triggered execution mode, the system stops processing after successfully refreshing the source table in the pipeline once, ensuring the table is updated based on the data available when the update started.
        # CONTINUOUS: If the pipeline uses continuous execution, the pipeline processes new data as it arrives in the source table to keep vector index fresh.
        "pipeline_type": "TRIGGERED",
    },
    # Embedding model to use
    # Tested configurations are available in the `supported_configs/embedding_models` Notebook
    # TODO: Replace with GTE that will be launched on FMAPI
    "embedding_config": {
        # Model Serving endpoint name
        "embedding_endpoint_name": "rag_demo_final_outputs_gte-large-en-v1_5",
        "embedding_tokenizer": {
            # Name of the embedding model that the tokenizer recognizes
            "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
            # Name of the tokenizer, either `hugging_face` or `tiktoken`
            "tokenizer_source": "hugging_face",
        },
    },
    # Parsing and chunking configuration
    # Changing this configuration here will NOT impact your data pipeline, these values are hardcoded in the POC data pipeline.
    # It is provided so you can copy / paste this configuration directly into the `Improve RAG quality` step and replicate the POC's data pipeline configuration
    "pipeline_config": {
        # File format of the source documents
        "file_format": "html",
        # Parser to use (must be present in `parser_library` Notebook)
        "parser": {"name": "html_to_markdown", "config": {}},
        # Chunker to use (must be present in `chunker_library` Notebook)
        "chunker": {
            "name": "langchain_recursive_char",
            "config": {
                "chunk_size_tokens": 1500,
                "chunk_overlap_tokens": 250,
            },
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output tables
# MAGIC
# MAGIC Next, we configure the output Delta Tables and Vector Index where the data pipeline will write the parsed/chunked/embedded data.

# COMMAND ----------

# Names of the output Delta Tables tables & Vector Search index
destination_tables_config = {
    # Staging table with the raw files & metadata
    "raw_files_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}_poc_raw_files_bronze",
    # Parsed documents
    "parsed_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}_poc_parsed_docs_silver",
    # Chunked documents that are loaded into the Vector Index
    "chunked_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}_poc_chunked_docs_gold",
    # Destination Vector Index
    "vectorsearch_index_name": f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}_poc_chunked_docs_gold_index",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load config
# MAGIC
# MAGIC This step loads the configuration so that the `02_poc_data_preperation` Notebook can use it.

# COMMAND ----------

import json

vectorsearch_config = data_pipeline_config['vectorsearch_config']
embedding_config = data_pipeline_config['embedding_config']
pipeline_config = data_pipeline_config['pipeline_config']

print(f"Using POC data pipeline config: {json.dumps(data_pipeline_config, indent=4)}\n")
print(f"Writing to: {json.dumps(destination_tables_config, indent=4)}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Chain config
# MAGIC
# MAGIC Next, we configure the chain's default settings.  The chain's code, stored in `single_turn_rag_chain` have been parameterized to use these variables.

# COMMAND ----------

# Notebook with the chain's code 
CHAIN_CODE_FILE = "single_turn_rag_chain"

# Chain configuration
rag_chain_config = {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
        # Databricks Model Serving endpoint name
        "llm_endpoint_name": "databricks-dbrx-instruct",
    },
    "retriever_config": {
        "vector_search_index": destination_tables_config["vectorsearch_index_name"],
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "chunked_text",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "path",
        },
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "ann"
        },
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.
Question: {question}
Context: {context}
Answer:""".strip(),
        "llm_prompt_template_variables": ["context", "question"],
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "Sample user question",
            }
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config & save to YAML
# MAGIC
# MAGIC This step saves the configuration so that the `03_deploy_poc_to_review_app` Notebook can use it.

# COMMAND ----------

import yaml
print(f"Using chain config: {json.dumps(rag_chain_config, indent=4)}\n")


with open('rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)

# COMMAND ----------

# MAGIC %md ## Load shared utilities used by the other notebooks

# COMMAND ----------

# MAGIC %run ./shared_utilities
