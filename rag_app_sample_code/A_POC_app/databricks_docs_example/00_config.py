# Databricks notebook source
# MAGIC %md # POC configuration

# COMMAND ----------

# MAGIC %pip install -U -qqqq install pyyaml mlflow mlflow-skinny

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %run ../../00_global_config

# COMMAND ----------

print(f"POC app using the UC catalog/schema {UC_CATALOG}.{UC_SCHEMA} with source data from {SOURCE_PATH} synced to the Vector Search endpoint {VECTOR_SEARCH_ENDPOINT}.  \n\nChain model will be logged to UC as {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}.  \n\nUsing MLflow Experiment `{MLFLOW_EXPERIMENT_NAME}` with data pipeline run name `{POC_DATA_PIPELINE_RUN_NAME}` and chain run name `{POC_CHAIN_RUN_NAME}`")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # POC Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation
# MAGIC ### Config
# MAGIC
# MAGIC Databricks reccomends starting with the default settings below for your POC.  Once you have collected stakeholder feedback, you will iterate on the app's quality using these parameters.
# MAGIC
# MAGIC To learn more about these settings, visit [link to guide].
# MAGIC
# MAGIC By default, we use [GTE Embeddings](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#call-a-bge-embeddings-model-using-databricks-model-serving-notebook) that is available on [Databricks Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html).  GTE is a high quality open source embedding model with a large context window.  We have selected a tokenizer and chunk size that matches this embedding model.

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
    # Changing this configuration here will change in the parser or chunker logic changing, becuase these functions are hardcoded in the POC data pipeline.  However, the configuration here does impact those functions.
    # It is provided so you can copy / paste this configuration directly into the `Improve RAG quality` step and replicate the POC's data pipeline configuration
    "pipeline_config": {
        # File format of the source documents
        "file_format": "json",
        # Parser to use (must be present in `parser_library` Notebook)
        "parser": {
            "name": "json",
            "config": {
                # The key of the JSON file that contains the content that should be chunked
                # All other keys will be passed through
                "content_key": "html_content"
            },
        },
        # Chunker to use (must be present in `chunker_library` Notebook)
        "chunker": {
            "name": "langchain_recursive_char",
            "config": {
                "chunk_size_tokens": 1024,
                "chunk_overlap_tokens": 256,
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
# MAGIC Next, we configure the chain's default settings.  The chain's code has been parameterized to use these variables. 
# MAGIC
# MAGIC Again, Databricks reccomends starting with the default settings below for your POC.  Once you have collected stakeholder feedback, you will iterate on the app's quality using these parameters.
# MAGIC
# MAGIC By default, we use `databricks-dbrx-instruct` but you can change this to any LLM hosted using Databricks Model Serving, including Azure OpenAI / OpenAI models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single or multi turn conversation?
# MAGIC
# MAGIC Let's take a sample converastion:
# MAGIC
# MAGIC > User: What is RAG?<br/>
# MAGIC > Assistant: RAG is a technique...<br/>
# MAGIC > User: How do I do it?<br/>
# MAGIC
# MAGIC A multi-turn conversation chain allows the assistant to understand that *it* in *how do I do it?* refers to *RAG*.  A single-turn conversation chain would not understand this context, as it treats every request as a completely new question.
# MAGIC
# MAGIC Most RAG use cases are for multi-turn conversation, however, the additional step required to understand *how do I do it?* uses an LLM and thus adds a small amount of latency.

# COMMAND ----------

# Notebook with the chain's code.  Choose one based on your requirements.  
# If you are not sure, use the `multi_turn_rag_chain`.

# CHAIN_CODE_FILE = "single_turn_rag_chain"

CHAIN_CODE_FILE = "multi_turn_rag_chain"

# COMMAND ----------

# Chain configuration
# We suggest using these default settings
rag_chain_config = {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
        # Databricks Model Serving endpoint name
        # This is the generator LLM where your LLM queries are sent.
        "llm_endpoint_name": "databricks-dbrx-instruct",
    },
    "retriever_config": {
        # Vector Search index that is created by the data pipeline
        "vector_search_index": destination_tables_config["vectorsearch_index_name"],
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "chunked_text",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "url",
        },
        # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "ann",
        },
        # Tag for the data pipeline, allowing you to easily compare the POC results vs. future data pipeline configurations you try.
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
        "llm_system_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}""".strip(),
        # Parameters that control how the LLM responds.
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
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
print(f"Using chain config: {json.dumps(rag_chain_config, indent=4)}\n\n Using chain file: {CHAIN_CODE_FILE}")

with open('rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)

# COMMAND ----------

# MAGIC %md ## Load shared utilities used by the other notebooks

# COMMAND ----------

# MAGIC %run ../z_shared_utilities
