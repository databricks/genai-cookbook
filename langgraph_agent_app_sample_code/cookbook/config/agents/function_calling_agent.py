from typing import List, Any, Dict, Union
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

from cookbook.config.shared.tool import ToolConfig, VectorSearchToolConfig, \
    UCFunctionToolConfig


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
    tool_configs: List[Any]

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["tool_configs"] = [
            yaml.safe_load(serializable_config_to_yaml(tool)) for tool in self.tool_configs
        ]
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_dict in data["tool_configs"]:
            tool_yml = yaml.dump(tool_dict)
            tools.append(load_serializable_config_from_yaml(tool_yml))

        # Replace tools with deserialized instances
        data["tool_configs"] = tools
        return class_object(**data)
