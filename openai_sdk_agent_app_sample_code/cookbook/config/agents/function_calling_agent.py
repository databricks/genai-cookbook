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
from mlflow.models.resources import DatabricksResource, DatabricksServingEndpoint


class FunctionCallingAgentConfig(SerializableConfig):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        tools (List[BaseTool]): List of tools used by the agent.
    """

    tools: List[Any]
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

    # name: str
    # description: str
    # endpoint_name: str

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["tools"] = [
            yaml.safe_load(serializable_config_to_yaml(tool)) for tool in self.tools
        ]
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_dict in data["tools"]:
            tool_yml = yaml.dump(tool_dict)
            tools.append(load_serializable_config_from_yaml(tool_yml))

        # Replace tools with deserialized instances
        data["tools"] = tools
        return class_object(**data)

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        dependencies = [
            DatabricksServingEndpoint(endpoint_name=self.llm_config.llm_endpoint_name),
        ]

        # Add the Databricks resources for the retriever's vector indexes
        for tool in self.tools:
            dependencies.extend(tool.get_resource_dependencies())
        return dependencies
