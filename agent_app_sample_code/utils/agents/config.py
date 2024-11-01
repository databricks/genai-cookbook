from typing import List, Any
import yaml
from pydantic import BaseModel
from utils.pydantic_utils import load_obj_from_yaml, obj_to_yaml
from utils.agents.tools import BaseTool
from utils.agents.llm import LLMConfig
from utils.agents.vector_search import VectorSearchRetrieverConfig
import os
def load_first_yaml_file(config_paths: List[str]) -> str:
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as handle:
                return handle.read()
    raise ValueError(f"No config file found at any of the following paths: {config_paths}. "
                     f"Please ensure a config file exists at one of those paths.")

class AgentConfig(BaseModel):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        tools (List[BaseTool]): List of tools used by the agent.
    """
    tools: List[BaseTool]
    llm_config: LLMConfig
    # Used by MLflow to set the Agent's input schema
    input_example: Any

    def to_yaml(self) -> str:
        # Serialize tools with their class paths
        # exclude_none = True prevents unused parameters, such as additional LLM parameters, from being included in the config
        data = self.dict(exclude_none=True)
        data['tools'] = [
            yaml.safe_load(obj_to_yaml(tool))
            for tool in self.tools
        ]
        return yaml.dump(data, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'AgentConfig':
        # Load the data from YAML
        agent_config_dict = yaml.safe_load(yaml_str)
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_dict in agent_config_dict['tools']:
            tool_yml = yaml.dump(tool_dict)
            tools.append(load_obj_from_yaml(tool_yml))

        # Replace tools with deserialized instances
        agent_config_dict['tools'] = tools
        return cls(**agent_config_dict)


class RAGConfig(BaseModel):
    """
    Configuration for a RAG chain with MLflow input example.

    Attributes:
        llm_config (LLMConfig): Configuration for the function-calling LLM.
        vector_search_retriever_config (VectorSearchRetrieverConfig): Configuration for the Databricks vector search
        index.
        input_example (Any): Used by MLflow to set the RAG chain's input schema.
    """

    vector_search_retriever_config: VectorSearchRetrieverConfig
    llm_config: LLMConfig
    # Used by MLflow to set the Agent's input schema
    input_example: Any
