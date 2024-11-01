import os
import mlflow
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from databricks.vector_search.client import VectorSearchClient
from utils.agents.vector_search import (
    VectorSearchRetrieverTool,
)

from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)


def get_model_config_from_paths(paths):
    for path in paths:
        if os.path.exists(path):
            return mlflow.models.ModelConfig(development_config=path)
    raise Exception(f"No ModelConfig YAML file found in any of the following "
                    f"paths {paths}. Please ensure a valid YAML config for the "
                    f"agent exists at one of the above paths.")

def get_agent_dependencies(agent_config):
    dependencies = [
        DatabricksServingEndpoint(
            endpoint_name=agent_config.llm_config.llm_endpoint_name
        ),
    ]

    # Add the Databricks resources for the retriever's vector indexes
    for tool in agent_config.tools:
        dependencies.extend(tool.get_resource_dependencies())
    return dependencies

def get_rag_dependencies(rag_config):
    return [
        DatabricksServingEndpoint(
            endpoint_name=rag_config.llm_config.llm_endpoint_name
        ),
        DatabricksVectorSearchIndex(index_name=rag_config.vector_search_retriever_config.vector_search_index)
    ]

def _log_agent_helper(log_model_method, resource_dependencies, agent_definition_file_path, input_example):
    model_path = os.path.join(os.getcwd(), agent_definition_file_path)
    return log_model_method(
        model_path,
        artifact_path="agent",
        input_example=input_example,
        resources=resource_dependencies,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(), # TODO: Add in `messages` to signature
        ),
        code_paths=[os.path.join(os.getcwd(), "utils")],
    )

def log_langchain_agent(resource_dependencies, agent_definition_file_path, input_example):
    model_path = os.path.join(os.getcwd(), agent_definition_file_path)
    return mlflow.langchain.log_model(
        lc_model=model_path,
        artifact_path="agent",
        input_example=input_example,
        resources=resource_dependencies,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(), # TODO: Add in `messages` to signature
        ),
        code_paths=[os.path.join(os.getcwd(), "utils")],
    )

def log_pyfunc_agent(resource_dependencies, agent_definition_file_path, input_example):
    model_path = os.path.join(os.getcwd(), agent_definition_file_path)
    return mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=model_path,
        input_example=input_example,
        resources=resource_dependencies,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(), # TODO: Add in `messages` to signature
        ),
        code_paths=[os.path.join(os.getcwd(), "utils")],
    )