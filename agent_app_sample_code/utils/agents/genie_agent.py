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

    # def to_yaml(self) -> str:
    #     # exclude_none = True prevents unused parameters, such as additional LLM parameters, from being included in the config
    #     data = self.model_dump(exclude_none=True)
    #     return yaml.dump(data, default_flow_style=False)

    # @classmethod
    # def from_yaml(cls, yaml_str: str) -> "GenieAgentConfig":
    #     # Load the data from YAML
    #     agent_config_dict = yaml.safe_load(yaml_str)
    #     return cls(**agent_config_dict)
