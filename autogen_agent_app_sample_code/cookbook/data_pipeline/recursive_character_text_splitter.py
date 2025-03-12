from typing import Callable, Tuple, Optional
from databricks.sdk import WorkspaceClient
from pydantic import BaseModel

# %md
# ##### `get_recursive_character_text_splitter`

# `get_recursive_character_text_splitter` creates a new function that, given an embedding endpoint, returns a callable that can chunk text documents. This utility allows you to write the core business logic of the chunker, without dealing with the details of text splitting. You can decide to write your own, or edit this code if it does not fit your use case.

# **Arguments:**

# - `model_serving_endpoint`: The name of the Model Serving endpoint with the embedding model.
# - `embedding_model_name`: The name of the embedding model e.g., `gte-large-en-v1.5`, etc.   If `model_serving_endpoint` is an OpenAI External Model or FMAPI model and set to `None`, this will be automatically detected.
# - `chunk_size_tokens`: An optional size for each chunk in tokens. Defaults to `None`, which uses the model's entire context window.
# - `chunk_overlap_tokens`: Tokens that should overlap between chunks. Defaults to `0`.

# **Returns:** A callable that takes a document (`str`) and produces a list of chunks (`list[str]`).

# Constants
HF_CACHE_DIR = "/tmp/hf_cache/"

# Embedding Models Configuration
EMBEDDING_MODELS = {
    "gte-large-en-v1.5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "Alibaba-NLP/gte-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 8192,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge-large-en-v1.5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "bge_large_en_v1_5": {
        # "tokenizer": lambda: AutoTokenizer.from_pretrained(
        #     "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
        # ),
        "context_window": 512,
        "type": "SENTENCE_TRANSFORMER",
    },
    "text-embedding-ada-002": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-ada-002"),
        "type": "OPENAI",
    },
    "text-embedding-3-small": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-small"),
        "type": "OPENAI",
    },
    "text-embedding-3-large": {
        "context_window": 8192,
        # "tokenizer": lambda: tiktoken.encoding_for_model("text-embedding-3-large"),
        "type": "OPENAI",
    },
}


def get_workspace_client() -> WorkspaceClient:
    """Returns a WorkspaceClient instance."""
    return WorkspaceClient()


# TODO: this is a cheap hack to avoid importing tokenizer libs at the top level -  the datapipeline utils are imported by the agent notebook which won't have these libs loaded & we don't want to since autotokenizer is heavy weight.
def get_embedding_model_tokenizer(endpoint_type: str) -> Optional[dict]:
    from transformers import AutoTokenizer
    import tiktoken

    # copy here to prevent needing to install tokenizer libraries everywhere this is imported
    EMBEDDING_MODELS_W_TOKENIZER = {
        "gte-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "Alibaba-NLP/gte-large-en-v1.5", cache_dir=HF_CACHE_DIR
            ),
            "context_window": 8192,
            "type": "SENTENCE_TRANSFORMER",
        },
        "bge-large-en-v1.5": {
            "tokenizer": lambda: AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5", cache_dir=HF_CACHE_DIR
            ),
            "context_window": 512,
            "type": "SENTENCE_TRANSFORMER",
        },
        "bge_large_en_v1_5": {
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
    return EMBEDDING_MODELS_W_TOKENIZER.get(endpoint_type).get("tokenizer")


def get_embedding_model_config(endpoint_type: str) -> Optional[dict]:
    """
    Retrieve embedding model configuration by endpoint type.
    """

    return EMBEDDING_MODELS.get(endpoint_type)


def extract_endpoint_type(llm_endpoint) -> Optional[str]:
    """
    Extract the endpoint type from the given llm_endpoint object.
    """
    try:
        return llm_endpoint.config.served_entities[0].external_model.name
    except AttributeError:
        try:
            return llm_endpoint.config.served_entities[0].foundation_model.name
        except AttributeError:
            return None


def detect_fmapi_embedding_model_type(
    model_serving_endpoint: str,
) -> Tuple[Optional[str], Optional[dict]]:
    """
    Detects the embedding model type and configuration for the given endpoint.
    Returns a tuple of (endpoint_type, embedding_config) or (None, None) if not found.
    """
    client = get_workspace_client()

    try:
        llm_endpoint = client.serving_endpoints.get(name=model_serving_endpoint)
        endpoint_type = extract_endpoint_type(llm_endpoint)
    except Exception as e:
        endpoint_type = None

    embedding_config = (
        get_embedding_model_config(endpoint_type) if endpoint_type else None
    )

    embedding_config["tokenizer"] = (
        get_embedding_model_tokenizer(endpoint_type) if endpoint_type else None
    )

    return (endpoint_type, embedding_config)


def validate_chunk_size(chunk_spec: dict):
    """
    Validate the chunk size and overlap settings in chunk_spec.
    Raises ValueError if any condition is violated.
    """
    if (
        chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]
    ) > chunk_spec["context_window"]:
        msg = (
            f'Proposed chunk_size of {chunk_spec["chunk_size_tokens"]} + overlap of {chunk_spec["chunk_overlap_tokens"]} '
            f'is {chunk_spec["chunk_overlap_tokens"] + chunk_spec["chunk_size_tokens"]} which is greater than context '
            f'window of {chunk_spec["context_window"]} tokens.',
        )
        return (False, msg)
    elif chunk_spec["chunk_overlap_tokens"] > chunk_spec["chunk_size_tokens"]:
        msg = (
            f'Proposed `chunk_overlap_tokens` of {chunk_spec["chunk_overlap_tokens"]} is greater than the '
            f'`chunk_size_tokens` of {chunk_spec["chunk_size_tokens"]}. Reduce the size of `chunk_size_tokens`.',
        )
        return (False, msg)
    else:
        context_usage = (
            round(
                (chunk_spec["chunk_size_tokens"] + chunk_spec["chunk_overlap_tokens"])
                / chunk_spec["context_window"],
                2,
            )
            * 100
        )
        msg = f'Chunk size in tokens: {chunk_spec["chunk_size_tokens"]} and chunk overlap in tokens: {chunk_spec["chunk_overlap_tokens"]} are valid.  Using {round(context_usage, 2)}% ({chunk_spec["chunk_size_tokens"] + chunk_spec["chunk_overlap_tokens"]} tokens) of the {chunk_spec["context_window"]} token context window.'
        return (True, msg)


def get_recursive_character_text_splitter(
    model_serving_endpoint: str,
    embedding_model_name: str = None,
    chunk_size_tokens: int = None,
    chunk_overlap_tokens: int = 0,
) -> Callable[[str], list[str]]:
    # imports here to prevent needing to install everywhere

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer
    import tiktoken

    try:
        # Detect the embedding model and its configuration
        embedding_model_name, chunk_spec = detect_fmapi_embedding_model_type(
            model_serving_endpoint
        )

        if chunk_spec is None or embedding_model_name is None:
            # Fall back to using provided embedding_model_name
            chunk_spec = EMBEDDING_MODELS.get(embedding_model_name)
            if chunk_spec is None:
                raise KeyError

        # Update chunk specification based on provided parameters
        chunk_spec["chunk_size_tokens"] = (
            chunk_size_tokens or chunk_spec["context_window"]
        )
        chunk_spec["chunk_overlap_tokens"] = chunk_overlap_tokens

        # Validate chunk size and overlap
        is_valid, msg = validate_chunk_size(chunk_spec)
        if not is_valid:
            raise ValueError(msg)
        else:
            print(msg)

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
        elif chunk_spec["type"] == "OPENAI":
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer.name,
                chunk_size=chunk_spec["chunk_size_tokens"],
                chunk_overlap=chunk_spec["chunk_overlap_tokens"],
            )
        else:
            raise ValueError(f"Unsupported model type: {chunk_spec['type']}")
        return splitter.split_text(text)

    return _recursive_character_text_splitter
