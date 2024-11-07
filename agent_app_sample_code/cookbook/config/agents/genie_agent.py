from typing import Any, List
from cookbook.config import SerializableConfig
from mlflow.models.resources import DatabricksResource, DatabricksGenieSpace


class GenieAgentConfig(SerializableConfig):
    """
    Configuration for the agent with MLflow input example.

    Attributes:
        llm_config (FunctionCallingLLMConfig): Configuration for the function-calling LLM.
        input_example (Any): Used by MLflow to set the Agent's input schema.
    """

    # TODO: Add validation for the genie_space_id once the API is available.
    genie_space_id: str

    # Used by MLflow to set the Agent's input schema
    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "What types of data can I query?",
            },
        ]
    }

    encountered_error_user_message: str = (
        "I encountered an error trying to answer your question, please try again."
    )

    # name: str
    # description: str
    # endpoint_name: str

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        return [DatabricksGenieSpace(genie_space_id=self.genie_space_id)]
