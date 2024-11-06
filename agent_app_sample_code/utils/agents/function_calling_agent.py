from typing import List, Any, Dict
import yaml
import importlib
from pydantic import BaseModel
from utils.agents.tools import obj_to_yaml, _load_class_from_dict, load_obj_from_yaml
from utils.agents.tools import Tool
from utils.agents.llm import LLMConfig
from utils.agents.tools import SerializableModel


class FunctionCallingAgentConfig(SerializableModel):
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
    input_example: Any

    # TODO: override model_dump instead
    # def to_yaml(self) -> str:
    #     # Serialize tools with their class paths
    #     # exclude_none = True prevents unused parameters, such as additional LLM parameters, from being included in the config
    #     data = self.model_dump(exclude_none=True)
    #     data["tools"] = [yaml.safe_load(obj_to_yaml(tool)) for tool in self.tools]
    #     return yaml.dump(data, default_flow_style=False)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["tools"] = [
            yaml.safe_load(obj_to_yaml(tool)) for tool in self.tools
        ]
        return model_dumped

    # @classmethod
    # def from_yaml(cls, yaml_str: str) -> "FunctionCallingAgentConfig":
    #     # Load the data from YAML
    #     agent_config_dict = yaml.safe_load(yaml_str)
    #     # Deserialize tools, dynamically reconstructing each tool
    #     tools = []
    #     for tool_dict in agent_config_dict["tools"]:
    #         tool_yml = yaml.dump(tool_dict)
    #         tools.append(load_obj_from_yaml(tool_yml))

    #     # Replace tools with deserialized instances
    #     agent_config_dict["tools"] = tools
    #     return cls(**agent_config_dict)

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableModel":
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_dict in data["tools"]:
            tool_yml = yaml.dump(tool_dict)
            tools.append(load_obj_from_yaml(tool_yml))

        # Replace tools with deserialized instances
        data["tools"] = tools
        return class_object(**data)

    # @classmethod
    # def from_dict(cls, data: Dict[str, Any]) -> "SerializableModel":
    #     """Create an instance from a dictionary, handling class path for proper instantiation.

    #     Args:
    #         data: Dictionary containing the model data and class path

    #     Returns:
    #         SerializableModel: An instance of the appropriate subclass
    #     """
    #     if not isinstance(data, dict):
    #         raise ValueError(f"Expected dict, got {type(data)}")

    #     # class_path = data.pop(_CLASS_PATH_KEY)
    #     # # Dynamically import the module and class
    #     # module_name, class_name = class_path.rsplit(".", 1)
    #     # module = importlib.import_module(module_name)
    #     # class_obj = getattr(module, class_name)
    #     print("test")
    #     print(data)
    #     class_obj, remaining_data = _load_class_from_dict(data)

    #     # Deserialize tools, dynamically reconstructing each tool
    #     tools = []
    #     for tool_dict in data["tools"]:
    #         tool_yml = yaml.dump(tool_dict)
    #         tools.append(load_obj_from_yaml(tool_yml))

    #     # Replace tools with deserialized instances
    #     remaining_data["tools"] = tools

    #     # Instantiate the class with remaining data
    #     return class_obj(**remaining_data)
