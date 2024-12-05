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

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        dependencies = [
            DatabricksServingEndpoint(endpoint_name=self.llm_config.llm_endpoint_name),
            DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
            DatabricksVectorSearchIndex(
                index_name="shared.cookbook_local_test_udhay.test_product_docs_docs_chunked_index__v2"
            ),
            DatabricksFunction(
                function_name="shared.cookbook_local_test_udhay.python_exec"
            ),
        ]

        return dependencies