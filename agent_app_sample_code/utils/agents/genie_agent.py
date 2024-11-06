from pydantic import BaseModel
from typing import Any
import yaml

from utils.agents.tools import SerializableModel


class GenieAgentConfig(SerializableModel):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (FunctionCallingLLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
    """

    # TODO: Add validation for the genie_space_id once the API is available.
    genie_space_id: str

    # Used by MLflow to set the Agent's input schema
    input_example: Any

    encountered_error_user_message: str = (
        "I encountered an error trying to answer your question, please try again."
    )
