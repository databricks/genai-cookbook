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
from utils.agents.config import (
    AgentConfig,
    LLMConfig,
    LLMParametersConfig,
    RetrieverConfig,
    RetrieverParametersConfig,
    RetrieverSchemaConfig,
)

@mlflow.trace(span_type="FUNCTION")
def execute_function(tool_functions, function_name, args):
    the_function = tool_functions[function_name]

    result = the_function(**args)
    return result

def chat_completion(model_serving_client, llm_endpoint_name, llm_parameters, messages: List[Dict[str, str]]):
    traced_create = mlflow.trace(
        model_serving_client.predict,
        name="chat_completions_api",
        span_type="CHAT_MODEL",
    )

    inputs = {
        "messages": messages,
        **llm_parameters,
    }

    # Use the traced_create to make the prediction
    return traced_create(
        endpoint=llm_endpoint_name,
        inputs=inputs,
    )

@mlflow.trace(span_type="PARSER")
def get_messages_array(model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]) -> List[Dict[str, str]]:
    if type(model_input) == ChatCompletionRequest:
        return model_input.messages
    elif type(model_input) == dict:
        return model_input.get("messages")
    elif type(model_input) == pd.DataFrame:
        return model_input.iloc[0].to_dict().get("messages")

@mlflow.trace(span_type="PARSER")
def extract_user_query_string(chat_messages_array: List[Dict[str, str]]) -> str:
    """
    Extracts user query string from the chat messages array.

    Args:
        chat_messages_array: Array of chat messages.

    Returns:
        User query string.
    """

    if isinstance(chat_messages_array, pd.Series):
        chat_messages_array = chat_messages_array.tolist()

    if isinstance(chat_messages_array[-1], dict):
        return chat_messages_array[-1]["content"]
    elif isinstance(chat_messages_array[-1], Message):
        return chat_messages_array[-1].content
    else:
        return chat_messages_array[-1]

@mlflow.trace(span_type="PARSER")
def extract_chat_history(chat_messages_array: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extracts the chat history from the chat messages array.

    Args:
        chat_messages_array: Array of chat messages.

    Returns:
        The chat history.
    """
    # Convert DataFrame to dict
    if isinstance(chat_messages_array, pd.Series):
        chat_messages_array = chat_messages_array.tolist()

    # Dictionary, return as is
    if isinstance(chat_messages_array[0], dict):
        return chat_messages_array[:-1]  # return all messages except the last one
    # MLflow Message, convert to Dictionary
    elif isinstance(chat_messages_array[0], Message):
        new_array = []
        for message in chat_messages_array[:-1]:
            new_array.append(asdict(message))
        return new_array
    else:
        raise ValueError(
            "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
        )

def log_agent_to_mlflow(agent_config, retriever_config):
    # Add the Databricks resources so that credentials are automatically provisioned by agents.deploy(...)
    databricks_resources = [
        DatabricksServingEndpoint(
            endpoint_name=agent_config.llm_config.llm_endpoint_name
        ),
    ]

    # Add the Databricks resources for the retriever's vector indexes
    for tool in agent_config.llm_config.tools:
        if type(tool) == RetrieverConfig:
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
    model_file = "agents/function_calling_agent/function_calling_agent_mlflow_sdk"
    model_path = os.path.join(os.getcwd(), model_file)

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