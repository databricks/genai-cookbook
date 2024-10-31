from typing miport List, Any
from pydantic import BaseModel
from utils.agents.pydantic_utils import BaseToolModel
from utils.agents.llm import LLMConfig
from utils.agents.vector_search import VectorSearchRetrieverConfig

class AgentConfig(BaseModel):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
    """

    tools: List[BaseToolModel]
    llm_config: LLMConfig

    # Used by MLflow to set the Agent's input schema
    input_example: Any

class RAGConfig(BaseModel):
    """
    Configuration for a RAG chain with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        vector_search_retriever_config (VectorSearchRetrieverConfig): Configuration for the Databricks vector search
        index.
        input_example (Any): Used by MLflow to set the RAG chain's input schema.
    """

    vector_search_retriever_config: VectorSearchRetrieverConfig
    llm_config: LLMConfig

    # Used by MLflow to set the Agent's input schema
    input_example: Any
