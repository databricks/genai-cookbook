import mlflow
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message


def chat_completion(
    model_serving_client,
    llm_endpoint_name,
    llm_parameters,
    messages: List[Dict[str, str]],
):
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
def get_messages_array(
    model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
) -> List[Dict[str, str]]:
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
def extract_chat_history(
    chat_messages_array: List[Dict[str, str]]
) -> List[Dict[str, str]]:
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


@mlflow.trace(span_type="PARSER")
def convert_messages_to_open_ai_format(
    chat_messages_array: List[Dict[str, str]]
) -> List[Dict[str, str]]:
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
        return chat_messages_array  # return all messages except the last one
    # MLflow Message, convert to Dictionary
    elif isinstance(chat_messages_array[0], Message):
        new_array = []
        for message in chat_messages_array:
            new_array.append(asdict(message))
        return new_array
    else:
        raise ValueError(
            "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
        )


@mlflow.trace(span_type="PARSER")
def concat_messages_array_to_string(messages):
    concatenated_message = "\n".join(
        [
            (
                f"{message.get('role', message.get('name', 'unknown'))}: {message.get('content', '')}"
                if message.get("role") in ("assistant", "user")
                else ""
            )
            for message in messages
        ]
    )
    return concatenated_message
