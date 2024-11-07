from cookbook.config.base import SerializableConfig
from cookbook.databricks_utils import (
    get_workspace_hostname,
)
from cookbook.data_pipeline.recursive_character_text_splitter import (
    EMBEDDING_MODELS,
    detect_fmapi_embedding_model_type,
    validate_chunk_size,
)


from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.serving import EndpointStateReady


class RecursiveTextSplitterChunkingConfig(SerializableConfig):
    """
    Configuration for the Unstructured Data Pipeline.

    Args:
        embedding_model_endpoint (str):
            Embedding model endpoint hosted on Model Serving.  Default is `databricks-gte-large`.  This can be an External Model, such as OpenAI or a Databricks hosted model on Foundational Model API. The list of Databricks hosted models can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        chunk_size_tokens (int):
            The size of each chunk of the document in tokens. Default is 1024.
        chunk_overlap_tokens (int):
            The overlap of tokens between chunks. Default is 256.
    """

    embedding_model_endpoint: str = "databricks-gte-large-en"
    chunk_size_tokens: int = 1024
    chunk_overlap_tokens: int = 256

    def validate_embedding_endpoint(self) -> tuple[bool, str]:
        """
        Validates that the specified embedding endpoint exists and is of the correct type
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        task_type = "llm/v1/embeddings"
        w = WorkspaceClient()
        browser_url = get_workspace_hostname()
        try:
            llm_endpoint = w.serving_endpoints.get(name=self.embedding_model_endpoint)
        except ResourceDoesNotExist as e:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} not found."
            return (False, msg)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} is not in a READY state.  Please visit the status page to debug: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
            return (False, msg)
        if llm_endpoint.task != task_type:
            msg = f"Model serving endpoint {self.embedding_model_endpoint} is online & ready, but does not support task type {task_type}.  Details at: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
            return (False, msg)

        msg = f"Validated serving endpoint {self.embedding_model_endpoint} as READY and of type {task_type}.  View here: {browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
        print(msg)
        return (True, msg)

    def validate_chunk_size_and_overlap(self) -> tuple[bool, str]:
        """
        Validates that chunk_size and overlap values are valid
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        # Detect the embedding model and its configuration
        embedding_model_name, chunk_spec = detect_fmapi_embedding_model_type(
            self.embedding_model_endpoint
        )

        # Update chunk specification based on provided parameters
        chunk_spec["chunk_size_tokens"] = self.chunk_size_tokens
        chunk_spec["chunk_overlap_tokens"] = self.chunk_overlap_tokens

        if chunk_spec is None or embedding_model_name is None:
            # Fall back to using provided embedding_model_name
            chunk_spec = EMBEDDING_MODELS.get(embedding_model_name)
            if chunk_spec is None:
                msg = f"Embedding model `{embedding_model_name}` not found, so can't validate chunking config. Chunking config must be validated for a specific embedding model.  Available models: {EMBEDDING_MODELS.keys()}"
                return (False, msg)

        # Validate chunk size and overlap
        is_valid, msg = validate_chunk_size(chunk_spec)
        if not is_valid:
            return (False, msg)
        else:
            print(msg)
            return (True, msg)
