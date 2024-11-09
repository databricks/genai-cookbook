from cookbook.tools import Tool
from cookbook.databricks_utils import get_function_url


import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from mlflow.models.resources import DatabricksFunction, DatabricksResource
from pydantic import Field, model_validator
from pyspark.errors import SparkRuntimeException
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.openai.toolkit import UCFunctionToolkit
from dataclasses import asdict

import json
from typing import Any, Dict, List


class UCTool(Tool):
    """Configuration for a Unity Catalog function tool.

    This class defines the configuration for a Unity Catalog function that can be used
    as a tool in an agent system.

    Args:
        uc_function_name: Unity Catalog location of the function in format: catalog.schema.function_name.
            Example: my_catalog.my_schema.my_function

    Returns:
        UCTool: A configured Unity Catalog function tool object.
    """

    uc_function_name: str
    """Unity Catalog location of the function in format: catalog.schema.function_name."""

    error_prompt: str = (
        "Error in generated code.  Please think step-by-step about how to fix the error and try calling this tool again with corrected inputs that reflect this thinking."
    )

    # Optional b/c we set these automatically in model_post_init from the UC function itself.
    # Suggest not overriding these, but rather updating the UC function's metadata directly.
    name: str = Field(default=None)  # Make it optional in the constructor
    description: str = Field(default=None)  # Make it optional in the constructor

    @model_validator(mode="after")
    def validate_uc_function_name(self) -> "UCTool":
        """Validates that the UC function exists and is accessible.

        Checks that the function name is properly formatted and exists in Unity Catalog
        with proper permissions.

        Returns:
            UCTool: The validated tool instance.

        Raises:
            ValueError: If function name is invalid or function is not accessible.
        """
        parts = self.uc_function_name.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"uc_function_name must be in format: catalog.schema.function_name; got `{self.uc_function_name}`"
            )

        # Validate that the function exists in Unity Catalog & user has EXECUTE permission on the function
        # Docs: https://databricks-sdk-py.readthedocs.io/en/stable/workspace/catalog/functions.html#get
        w = WorkspaceClient()
        try:
            w.functions.get(name=self.uc_function_name)
        except ResourceDoesNotExist:
            raise ValueError(
                f"Function `{self.uc_function_name}` not found in Unity Catalog or you do not have permission to access it.  Ensure the function exists, and you have EXECUTE permission on the function, USE CATALOG and USE SCHEMA permissions on the catalog and schema.  If function exists, you can verify permissions here: {get_function_url(self.uc_function_name)}."
            )

        return self

    def model_post_init(self, __context: Any) -> None:

        # Initialize the UC clients
        self._uc_client = DatabricksFunctionClient()
        self._toolkit = UCFunctionToolkit(
            function_names=[self.uc_function_name], client=self._uc_client
        )

        # OK to use [0] position b/c we know that there is only one function initialized in the toolkit.
        self.name = self._toolkit.tools[0]["function"]["name"]
        self.description = self._toolkit.tools[0]["function"]["description"]

    def _get_parameters_schema(self) -> dict:
        """Gets the parameter schema for the UC function.

        Returns:
            dict: JSON schema describing the function's parameters.
        """
        # OK to use [0] position b/c we know that there is only one function initialized in the toolkit.
        return self._toolkit.tools[0]["function"]["parameters"]

    @mlflow.trace(span_type="TOOL", name="uc_tool")
    def __call__(self, **kwargs) -> Dict[str, str]:
        # annotate the span with the tool name
        span = mlflow.get_current_active_span()
        span.set_attributes({"uc_tool_name": self.uc_function_name})

        # trace the function call
        traced_exec_function = mlflow.trace(
            span_type="FUNCTION", name="_uc_client.execute_function"
        )(self._uc_client.execute_function)

        # convert input args to json
        args_json = json.loads(json.dumps(kwargs, default=str))

        # TODO: Add in Ben's code parser

        try:
            result = traced_exec_function(
                function_name=self.uc_function_name, parameters=args_json
            )
            return asdict(result)

        # Parse the error into a format that's easier for the LLM to understand w/ out any of the Spark runtime error noise
        except SparkRuntimeException as e:
            try:
                error = (
                    e.getMessageParameters()["error"]
                    .replace('File "<string>",', "")
                    .strip()
                )
            except Exception as e:
                error = e.getMessageParameters()["error"]
            return {
                "status": self.error_prompt,
                "error": error,
            }
        except Exception as e:
            return {
                "status": self.error_prompt,
                "error": str(e),
            }

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        kwargs["exclude"] = {"name", "description"}.union(kwargs.get("exclude", set()))
        return super().model_dump(**kwargs)

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        return [DatabricksFunction(function_name=self.uc_function_name)]
