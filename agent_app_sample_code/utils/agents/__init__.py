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


# TODO(smurching) remove the default for `agent_definition_file_path`
def log_agent_to_mlflow(agent_config, retriever_config, agent_definition_file_path="agents/function_calling_agent/function_calling_agent_mlflow_sdk"):
    # Add the Databricks resources so that credentials are automatically provisioned by agents.deploy(...)
    databricks_resources = [
        DatabricksServingEndpoint(
            endpoint_name=agent_config.llm_config.llm_endpoint_name
        ),
    ]

    # Add the Databricks resources for the retriever's vector indexes
    for tool in agent_config.llm_config.tools:
        # TODO(smurching) fix this
        if type(tool) == VectorSearchRetrieverConfig:
            databricks_resources.append(
                DatabricksVectorSearchIndex(index_name=tool.vector_search_index)
            )
            index_embedding_model = (
                VectorSearchClient(disable_notice=True)
                .get_index(index_name=retriever_config.vector_search_index)
                .describe()
                .get("delta_sync_index_spec")
                .get("embedding_source_columns")[0]
                .get("embedding_model_endpoint_name")
            )
            if index_embedding_model is not None:
                databricks_resources.append(
                    DatabricksServingEndpoint(endpoint_name=index_embedding_model),
                )
            else:
                raise Exception("Could not identify the embedding model endpoint resource for {tool.vector_search_index}.  Please manually add the embedding model endpoint to `databricks_resources`.")

    # Specify the full path to the Agent notebook
    model_path = os.path.join(os.getcwd(), agent_definition_file_path)

    # Log the agent as an MLflow model
    return mlflow.pyfunc.log_model(
        python_model=model_path,
        model_config=agent_config.dict(),
        artifact_path="agent",
        input_example=agent_config.input_example,
        resources=databricks_resources,
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(), # TODO: Add in `messages` to signature
        ),
        code_paths=[os.path.join(os.getcwd(), "utils")],
    )
