# Databricks notebook source
# packages if running this notebook alone
# %pip install -U -qqq transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0  databricks-sdk

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `get_recursive_character_text_splitter`
# MAGIC
# MAGIC `get_recursive_character_text_splitter` creates a new function that, given an embedding endpoint, returns a callable that can chunk text documents. This utility allows you to write the core business logic of the chunker, without dealing with the details of text splitting. You can decide to write your own, or edit this code if it does not fit your use case.
# MAGIC
# MAGIC **Arguments:**
# MAGIC
# MAGIC - `model_serving_endpoint`: The name of the Model Serving endpoint with the embedding model.
# MAGIC - `embedding_model_name`: The name of the embedding model e.g., `gte-large-en-v1.5`, etc.   If `model_serving_endpoint` is an OpenAI External Model or FMAPI model and set to `None`, this will be automatically detected. 
# MAGIC - `chunk_size_tokens`: An optional size for each chunk in tokens. Defaults to `None`, which uses the model's entire context window.
# MAGIC - `chunk_overlap_tokens`: Tokens that should overlap between chunks. Defaults to `0`.
# MAGIC
# MAGIC **Returns:** A callable that takes a document (`str`) and produces a list of chunks (`list[str]`).

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import tiktoken
from typing import Callable
import os
from databricks.sdk import WorkspaceClient


# This is necessary because spark sometimes will give a warning about the cache in `/.cache` being read-only.
HF_CACHE_DIR = "/tmp/hf_cache/"


EMBEDDING_MODELS = {
    "gte-large-en-v1.5": {
        "tokenizer": lambda: AutoTokenizer.from_pretrained(
            "Alibaba-NLP/gte-large-en-v1.5", cache_dir=HF_CACHE_DIR
        ),
        "context_window": 8192,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge-large-en-v1.5": { # FMAPI Per-Token name
        "tokenizer": lambda: AutoTokenizer.from_pretrained(
            "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge_large_en_v1_5": { # FMAPI PT name
        "tokenizer": lambda: AutoTokenizer.from_pretrained(
            "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "text-embedding-ada-002": {
        "context_window": 8192,
        "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-ada-002"),
        "type": "OPENAI",
    },
    "text-embedding-3-small": {
        "context_window": 8192,
        "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-small"),
        "type": "OPENAI",
    },
    "text-embedding-3-large": {
        "context_window": 8192,
        "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-large"),
        "type": "OPENAI",
    },
}


def detect_fmapi_embedding_model_type(model_serving_endpoint):
    """
    Try to detect the endpoint type and get the embedding config
    returns (None, None) if not found
    returns (endpoint_type, embedding_config)
    """
    w = WorkspaceClient()

    endpoint_type = None
    try:
        llm_endpoint = w.serving_endpoints.get(name=model_serving_endpoint)
        # external model name
        endpoint_type = llm_endpoint.config.served_entities[0].external_model.name
    except Exception as e:
        try:
            # FMAPI pay per token
            endpoint_type = llm_endpoint.config.served_entities[0].foundation_model.name
        except Exception as e:
            try: 
                # FMAPI provisioned throughput
                endpoint_type = llm_endpoint.config.served_entities[0].name
                if endpoint_type[-2:][0] == '-':
                    # remove the version number if present e.g., `bge_large_en_v1_5-2` --> `bge_large_en_v1_5`
                    endpoint_type = endpoint_type[:-2]
            except Exception as e:
                pass
                

    if endpoint_type is not None:
        # will return None if not found
        embedding_config = EMBEDDING_MODELS.get(endpoint_type)
        if embedding_config is not None:
            return (endpoint_type, embedding_config)
        else:
            return None, None
    else:
        return None, None


def get_recursive_character_text_splitter(
    model_serving_endpoint: str,
    embedding_model_name: str = None,
    chunk_size_tokens: int = None,
    chunk_overlap_tokens: int = 0,
) -> Callable[[str], list[str]]:
    try:
        (embedding_model_name, chunk_spec) = detect_fmapi_embedding_model_type(
            model_serving_endpoint
        )
        if chunk_spec is not None and embedding_model_name is not None:
            print(
                f"Detected endpoint `{model_serving_endpoint}` as embedding model `{embedding_model_name}`"
            )
        else:
            # try to look it up based on the based type
            chunk_spec = EMBEDDING_MODELS[embedding_model_name]

        if chunk_size_tokens is not None:
            chunk_spec["chunk_size_tokens"] = chunk_size_tokens
        else:
            chunk_spec["chunk_size_tokens"] = chunk_spec["context_window"]

        chunk_spec["chunk_overlap_tokens"] = chunk_overlap_tokens

        if (
            chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]
        ) > chunk_spec["context_window"]:
            raise ValueError(
                f'Proposed chunk_size of {chunk_spec["chunk_size_tokens"]} + overlap of {chunk_spec["chunk_overlap_tokens"]} is {chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]} which is greater than context window of {chunk_spec["context_window"]} tokens'
            )

        if chunk_spec["chunk_overlap_tokens"] > chunk_spec["chunk_size_tokens"]:
            raise ValueError(
                f'Proposed `chunk_overlap_tokens` of {chunk_spec["chunk_overlap_tokens"]} is greater than the `chunk_size_tokens` of {chunk_spec["chunk_size_tokens"]}.  Reduce the size of `chunk_size_tokens`'
            )

        print(f'Chunk size in tokens: {chunk_spec["chunk_size_tokens"]}')
        print(f'Chunk overlap in tokens: {chunk_spec["chunk_overlap_tokens"]}')
      
        print(
            f'Using {round((chunk_spec["chunk_size_tokens"] + chunk_spec["chunk_overlap_tokens"])/chunk_spec["context_window"], 2)*100}% of the {chunk_spec["context_window"]} token context window.'
        )
        # print(chunk_spec)
    except KeyError:
        raise ValueError(
            f"Embedding model `{embedding_model_name}` not found. Available models: {EMBEDDING_MODELS.keys()}"
        )

    def _recursive_character_text_splitter(text: str) -> list[str]:
        tokenizer = chunk_spec["tokenizer"]()
        if chunk_spec["type"] == "SENTENCE_TRANSFORMER":
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
            )
            return splitter.split_text(text)
        elif chunk_spec["type"] == "OPENAI":
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer.name,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
            )
            return splitter.split_text(text)

    return _recursive_character_text_splitter
