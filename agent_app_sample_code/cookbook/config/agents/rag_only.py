from cookbook.config.common.llm import LLMConfig
from cookbook.config.tools.vector_search_tool import VectorSearchRetrieverTool


from pydantic import BaseModel


from typing import Any


class RAGConfig(BaseModel):
    """
    Configuration for a RAG chain with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        vector_search_retriever_config (VectorSearchRetrieverConfig): Configuration for the Databricks vector search
        index.
        input_example (Any): Used by MLflow to set the RAG chain's input schema.
    """

    vector_search_retriever_config: VectorSearchRetrieverTool
    llm_config: LLMConfig
    # Used by MLflow to set the Agent's input schema
    input_example: Any
