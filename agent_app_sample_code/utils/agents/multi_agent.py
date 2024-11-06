from pydantic import BaseModel
from typing import Any, List, Literal, Dict
from utils.agents.llm import LLMConfig
from utils.agents.tools import (
    SerializableModel,
    load_obj_from_yaml,
    _CLASS_PATH_KEY,
    obj_to_yaml,
)
import yaml


class MultiAgentConfig(SerializableModel):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        playground_debug_mode (bool): Flag to enable debug mode in playground. Defaults to False.
        agents (List[Any]): List of agents to be used in the multi-agent setup.
    """

    llm_config: LLMConfig

    # Used by MLflow to set the Agent's input schema
    input_example: Any

    playground_debug_mode: bool = False

    agent_loading_mode: Literal["local", "model_serving"] = "local"

    agents: List[Any]

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableModel":
        # Deserialize tools, dynamically reconstructing each tool
        agents = []
        for agent_dict in data["agents"]:
            agent_yml = yaml.dump(agent_dict)
            agents.append(load_obj_from_yaml(agent_yml))

        # Replace tools with deserialized instances
        data["agents"] = agents
        return class_object(**data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["agents"] = [
            yaml.safe_load(obj_to_yaml(agent)) for agent in self.agents
        ]
        return model_dumped
