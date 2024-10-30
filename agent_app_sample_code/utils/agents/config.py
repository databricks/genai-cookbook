from pydantic import computed_field, Field, BaseModel
from typing import Literal, List, Any


class ToolConfig(BaseModel):
    # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
    tool_description_prompt: str

    # The name of the tool.  Used by the Agent in conjunction with tool_description_prompt to determine if this tool is relevant to the query.
    tool_name: str

    # Input schema as a Pydantic BaseModel that will be converted to a json schema and passed to the LLM
    # tool_input_schema: dict

    # Name of the class within the Agent's code that will be instantiated and then called when the tool is selected by the LLM
    tool_class_name: str

    @property
    def tool_input_schema(self) -> dict:
        return {}

    @computed_field
    def tool_input_json_schema(self) -> dict:
        tool_input_json_schema = self.tool_input_schema
        # del tool_input_json_schema["title"]
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description_prompt,
                "parameters": tool_input_json_schema,
            },
        }


class RetrieverSchemaConfig(BaseModel):
    """
    Configuration for the schema used in the retriever's response.

    Attributes:
        primary_key (str): The column name in the retriever's response referred to the unique key.
            If using Databricks vector search with delta sync, this should be the column of the delta table that acts as the primary key.
            Example: "chunk_id" - The column name in the retriever's response referred to the unique key.
        chunk_text (str): The column name in the retriever's response that contains the returned chunk.
            Example: "content_chunked" - The column name in the retriever's response that contains the returned chunk.
        document_uri (str): The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            Example: "doc_uri" - The URI of the chunk - displayed as the document ID in the Review App.
        additional_metadata_columns (List[str]): Additional metadata columns to present to the LLM.
            Example: [] - Additional columns to return from the vector database and present to the LLM.
    """

    # The column name in the retriever's response referred to the unique key
    # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
    primary_key: str
    # The column name in the retriever's response that contains the returned chunk.
    chunk_text: str
    # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
    document_uri: str
    # Additional metadata columns to present to the LLM.
    additional_metadata_columns: List[str]


class RetrieverParametersConfig(BaseModel):
    """
    Configuration for the parameters used in the retriever.

    Attributes:
        num_results (int): The number of chunks to return for each query.
            Example: 5 - Number of search results that the retriever returns.
        query_type (Literal['ann', 'hybrid']): The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid` (keyword + semantic similarity).
            Example: "ann" - Type of search: ann or hybrid.
    """

    # The number of chunks to return for each query.
    num_results: int
    # The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid`
    # (keyword + semantic similarity)
    query_type: Literal["ann", "hybrid"]


class RetrieverConfig(ToolConfig):
    """
    Configuration for a retriever, which can be used either deterministically in a fixed RAG chain or
    as a tool.

    Attributes:
        vector_search_index (str): Vector Search index that is created by the data pipeline.
            Example: datapipeline_output_config.vector_index - UC Vector Search index.
        vector_search_schema (RetrieverSchemaConfig): Schema configuration for the retriever.
            Required by Agent Evaluation to:
            1. Enable the Review App to properly display retrieved chunks.
            2. Enable metrics / LLM judges to understand which fields to use to measure the retriever.
            Each is a column name within the `vector_search_index`.
        vector_search_threshold (float): Threshold for retrieved document similarity. Used to exclude results that are very dissimilar to the query.
            Example: 0.1 - 0 to 1, similarity threshold cut off for retrieved docs. Increase if the retriever is returning irrelevant content.
        chunk_template (str): Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.
            The f-string {chunk_text} and {metadata} can be used.
            Example: "Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n" - Prompt template used to format the retrieved information into {context} in `prompt_template`.
        prompt_template (str): Prompt template used to format all chunks for presentation to the LLM.
            The f-string {context} can be used.
        vector_search_parameters (RetrieverParametersConfig): Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
            Parameters defined by Vector Search docs: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#query-a-vector-search-endpoint
    """

    # Vector Search index that is created by the data pipeline
    vector_search_index: str

    vector_search_schema: RetrieverSchemaConfig

    # Threshold for retrieved document similarity.  Used to exclude results that are very dissimilar to the query.
    vector_search_threshold: float

    # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.  The f-string {chunk_text} and {metadata} can be used.
    chunk_template: str

    # Prompt template used to format all chunks for presentation to the LLM.  The f-string {context} can be used.
    prompt_template: str

    # Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
    vector_search_parameters: RetrieverParametersConfig

    retriever_query_parameter_prompt: str = "The query to find documents for."

    @property
    def tool_input_schema(self) -> dict:
        return {
            "properties": {
                "query": {
                    "default": None,
                    "description": self.retriever_query_parameter_prompt,
                    "type": "string",
                }
            },
            "type": "object",
        }


class LLMParametersConfig(BaseModel):
    """
    Configuration for LLM response parameters.

    Attributes:
        temperature (float): Controls randomness in the response.
        max_tokens (int): Maximum number of tokens in the response.
        top_p (float): Controls diversity via nucleus sampling.
        top_k (int): Limits the number of highest probability tokens considered.
    """

    # Parameters that control how the LLM responds.
    temperature: float = None
    max_tokens: int = None


class LLMConfig(BaseModel):
    """
    Configuration for the function-calling LLM.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name.
            This is the generator LLM where your LLM queries are sent.
            Databricks foundational model endpoints can be found here:
            https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        llm_system_prompt_template (str): Template for the LLM prompt.
            This is how the RAG chain combines the user's question and the retrieved context.
        llm_parameters (LLMParametersConfig): Parameters that control how the LLM responds.
        tools (List[Any]): List of tools available for the LLM.
    """

    # Databricks Model Serving endpoint name
    # This is the generator LLM where your LLM queries are sent.
    # Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    llm_endpoint_name: str

    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template: str

    # Parameters that control how the LLM responds.
    llm_parameters: LLMParametersConfig

    tools: List[Any] = []


class AgentConfig(BaseModel):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
    """

    llm_config: LLMConfig

    # Used by MLflow to set the Agent's input schema
    input_example: Any