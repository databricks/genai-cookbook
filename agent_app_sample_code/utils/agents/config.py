from typing import List, Any
import yaml
from pydantic import BaseModel
from utils.pydantic_utils import load_obj_from_yaml, obj_to_yaml
from utils.agents.tools import BaseToolModel
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
        tools (List[BaseToolModel]): List of tools used by the agent.
    """
    tools: List[BaseToolModel]
    llm_config: LLMConfig
    # Used by MLflow to set the Agent's input schema
    input_example: Any

    def to_yaml(self) -> str:
        # Serialize tools with their class paths
        data = self.dict()
        data['tools'] = [
            obj_to_yaml(tool)
            for tool in self.tools
        ]
        return yaml.dump(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'AgentConfig':
        # Load the data from YAML
        agent_config_dict = yaml.safe_load(yaml_str)
        # Deserialize tools, dynamically reconstructing each tool
        tools = []
        for tool_data in agent_config_dict['tools']:
            tools.append(load_obj_from_yaml(tool_data))

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
