from typing import List, Any
import yaml
from pydantic import BaseModel
from utils.agents.tools import obj_to_yaml
from utils.agents.tools import load_obj_from_yaml
from utils.agents.tools import Tool
from utils.agents.llm import LLMConfig


class FunctionCallingAgentConfig(BaseModel):
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

    def to_yaml(self) -> str:
        # Serialize tools with their class paths
        # exclude_none = True prevents unused parameters, such as additional LLM parameters, from being included in the config
        data = self.model_dump(exclude_none=True)
        data["tools"] = [yaml.safe_load(obj_to_yaml(tool)) for tool in self.tools]
        return yaml.dump(data, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FunctionCallingAgentConfig":
        # Load the data from YAML
        agent_config_dict = yaml.safe_load(yaml_str)
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_dict in agent_config_dict["tools"]:
            tool_yml = yaml.dump(tool_dict)
            tools.append(load_obj_from_yaml(tool_yml))

        # Replace tools with deserialized instances
        agent_config_dict["tools"] = tools
        return cls(**agent_config_dict)
