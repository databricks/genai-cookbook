# Databricks notebook source
# MAGIC %md
# MAGIC ##### AgentConfig
# MAGIC
# MAGIC `AgentConfig` is a configuration object that we use to communicate between our Agent notebook and the RAG chain that we log with mlflow, and also logged with mlflow. This configuration can be changed in conjunction with editing the chain file if parameters need to be edited.

# COMMAND ----------

# If you want to use this outside the context of the genai cookbook.
# %pip install pydantic

# COMMAND ----------

from pydantic import BaseModel
from typing import Literal, Any, List
import yaml
import os

# COMMAND ----------


class RetrieverSchemaConfig(BaseModel):
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
  # The number of chunks to return for each query.
  num_results: int
  # The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid`
  # (keyword + semantic similarity)
  query_type: Literal['ann', 'hybrid']

class RetrieverToolConfig(BaseModel):
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
  parameters: RetrieverParametersConfig

  # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
  tool_description_prompt: str

# COMMAND ----------



# NOTE: These configs are created to communicate between the core notebook, and the executed chain via `mlflow.log_model`. They are pydantic models, which are thin wrappers around Python dictionaries, used for validation of the config.


class LLMParametersConfig(BaseModel):
  # Parameters that control how the LLM responds.
  temperature: float
  max_tokens: int

class LLMConfig(BaseModel):
  # Databricks Model Serving endpoint name
  # This is the generator LLM where your LLM queries are sent.
  # Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
  llm_endpoint_name: str

  # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
  llm_system_prompt_template: str
  # Parameters that control how the LLM responds.
  llm_parameters: LLMParametersConfig

class AgentConfig(BaseModel):
  retriever_tool: RetrieverToolConfig
  llm_config: LLMConfig
  input_example: Any

def validate_agent_config(config: dict) -> None:
  AgentConfig.parse_obj(config)

def save_agent_config(config: dict, file_path: str) -> None:
  # Ensure the directory exists
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'w') as file:
      yaml.dump(config, file)