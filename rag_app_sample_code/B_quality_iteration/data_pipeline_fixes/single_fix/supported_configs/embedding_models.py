# Databricks notebook source
# MAGIC %md
# MAGIC ## Embedding model tested configurations
# MAGIC
# MAGIC The below configurations can be swapped into the `embedding_config` parameter in `00_config` <br/><br/>
# MAGIC
# MAGIC ```
# MAGIC CONFIG_TO_RUN = "your_short_name" ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC configurations = {
# MAGIC     "your_short_name": { ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC         ...
# MAGIC
# MAGIC         # REPLACE THE EMBEDDING CONFIG HERE
# MAGIC         "embedding_config": {
# MAGIC             # Model Serving endpoint name
# MAGIC             "embedding_endpoint_name": "databricks-bge-large-en",
# MAGIC             "embedding_tokenizer": {
# MAGIC                 # Name of the embedding model that the tokenizer recognizes
# MAGIC                 "tokenizer_model_name": "BAAI/bge-large-en-v1.5",
# MAGIC                 # Name of the tokenizer, either `hugging_face` or `tiktoken`
# MAGIC                 "tokenizer_source": "hugging_face",
# MAGIC             },
# MAGIC         },
# MAGIC
# MAGIC         ...
# MAGIC     },
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pay-per-token Databricks Foundational Model APIs
# MAGIC
# MAGIC The following embedding models are available on the [Foundational Model API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#pay-per-token-foundation-model-apis)

# COMMAND ----------

# GTE Large

# TODO: Update with the FM API endpoint
embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "databricks-gte-large-en",
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "hugging_face",
    },
}

# BGE Large

embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "databricks-bge-large-en",
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "BAAI/bge-large-en-v1.5",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "hugging_face",
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sentence Transformer compatible open source models
# MAGIC
# MAGIC Using Model Serving Custom Models, you can use most Sentence Transformer compatible open source embedding models.
# MAGIC
# MAGIC To load a model to Model Serving, use the `/helpers/SentenceTransformer_Embedding_Model_Loader` Notebook in this repo.

# COMMAND ----------

# Example for GTE

embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "YOUR_ENDPOINT_NAME_HERE", # REPLACE WITH YOUR EXTERNAL MODEL ENDPOINT NAME
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "hugging_face",
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Azure OpenAI or OpenAI
# MAGIC
# MAGIC Using these configurations requires setting up an External Model using Model Serving.  
# MAGIC
# MAGIC
# MAGIC To create an External Model, either:
# MAGIC 1. Use the `/helpers/Create_OpenAI_External_Model` Notebook in this repo
# MAGIC 2. Follow the [External Models tutorial](https://docs.databricks.com/en/generative-ai/external-models/external-models-tutorial.html)

# COMMAND ----------

# text-embedding-ada-002

embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "text-embedding-ada-002", # REPLACE WITH YOUR EXTERNAL MODEL ENDPOINT NAME
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "text-embedding-ada-002",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "tiktoken",
    },
}

# text-embedding-large

embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "text-embedding-large", # REPLACE WITH YOUR EXTERNAL MODEL ENDPOINT NAME
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "text-embedding-large",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "tiktoken",
    },
}

# text-embedding-small

embedding_config = {
    # Model Serving endpoint name
    "embedding_endpoint_name": "text-embedding-small", # REPLACE WITH YOUR EXTERNAL MODEL ENDPOINT NAME
    "embedding_tokenizer": {
        # Name of the embedding model that the tokenizer recognizes
        "tokenizer_model_name": "text-embedding-small",
        # Name of the tokenizer, either `hugging_face` or `tiktoken`
        "tokenizer_source": "tiktoken",
    },
}
