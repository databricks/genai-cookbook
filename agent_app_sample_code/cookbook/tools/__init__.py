from cookbook.config.base import SerializableConfig
from mlflow.models.resources import DatabricksResource


from typing import Any, List


class Tool(SerializableConfig):
    """Base class for all tools"""

    def __call__(self, **kwargs) -> Any:
        """Execute the tool with validated inputs"""
        raise NotImplementedError(
            "__call__ must be implemented by Tool subclasses. This method should execute "
            "the tool's functionality with the provided validated inputs and return the result."
        )

    name: str
    description: str

    def get_json_schema(self) -> dict:
        """Returns an OpenAPI-compatible JSON schema for the tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema(),
            },
        }

    def _get_parameters_schema(self) -> dict:
        """Returns the JSON schema for the tool's parameters."""
        raise NotImplementedError(
            "_get_parameters_schema must be implemented by Tool subclasses. This method should "
            "return an OpenAPI-compatible JSON schema dict describing the tool's input parameters. "
            "The schema should include parameter names, types, descriptions, and any validation rules."
        )

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        """Returns a list of Databricks resources (mlflow.models.resources.* objects) that the tool uses.  Used to securely provision credentials for these resources when the tool is deployed to Model Serving."""
        raise NotImplementedError(
            "get_resource_dependencies must be implemented by Tool subclasses. This method should "
            "return a list of mlflow.models.resources.* objects that the tool depends on."
        )
