from typing import List, Any, Dict
from cookbook.config import serializable_config_to_yaml
import yaml
from pydantic import BaseModel
from cookbook.config import (
    load_serializable_config_from_yaml,
)
from cookbook.config.shared.llm import LLMConfig
from cookbook.config import (
    SerializableConfig,
)
from mlflow.models.resources import DatabricksResource, DatabricksServingEndpoint, \
    DatabricksVectorSearchIndex

from cookbook.config.shared.tool import ToolConfig


class FunctionCallingAgentConfig(SerializableConfig):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        tools (List[BaseTool]): List of tools used by the agent.
    """

    llm_config: LLMConfig
    # Used by MLflow to set the Agent's input schema
    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "What can you help me with?",
            },
        ]
    }
    tool_configs: List[ToolConfig]

    # name: str
    # description: str
    # endpoint_name: str
