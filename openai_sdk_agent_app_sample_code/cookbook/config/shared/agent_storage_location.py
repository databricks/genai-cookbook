from pydantic import (
    field_validator,
    FieldValidationInfo,
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    ResourceDoesNotExist,
    NotFound,
)
from pydantic import Field
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.errors import NotFound
from cookbook.config import SerializableConfig
from databricks.sdk import WorkspaceClient


class AgentStorageConfig(SerializableConfig):
    """
    Source data configuration for the Unstructured Data Pipeline. You can modify this class to add additional configuration settings.

    Args:
      uc_model_name (str):
        Required. Fully qualified name of the model in format: catalog.schema.model_name
      evaluation_set_uc_table (str):
        Required. Fully qualified name of the evaluation table in format: catalog.schema.table_name
    """

    uc_model_name: str = Field(..., min_length=1)
    evaluation_set_uc_table: str = Field(..., min_length=1)
    mlflow_experiment_name: str = Field(None)

    @field_validator("uc_model_name", "evaluation_set_uc_table")
    @classmethod
    def validate_uc_fqn_format(cls, v: str, info: FieldValidationInfo) -> str:
        if v.count(".") != 2:
            raise ValueError(
                f"{info.field_name} must be in format: catalog.schema.name"
            )
        return v

    @classmethod
    def escape_uc_fqn(cls, uc_fqn: str) -> str:
        """
        Escape the fully qualified name (FQN) for a Unity Catalog asset if it contains special characters.

        Args:
            uc_fqn (str): The fully qualified name of the asset.

        Returns:
            str: The escaped fully qualified name if it contains special characters, otherwise the original FQN.
        """
        if "-" in uc_fqn:
            parts = uc_fqn.split(".")
            escaped_parts = [f"`{part}`" for part in parts]
            return ".".join(escaped_parts)
        else:
            return uc_fqn

    def check_if_catalog_exists(self, catalog_name: str) -> bool:
        w = WorkspaceClient()
        try:
            w.catalogs.get(name=catalog_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def check_if_schema_exists(self, catalog_name: str, schema_name: str) -> bool:
        w = WorkspaceClient()
        try:
            full_name = f"{catalog_name}.{schema_name}"
            w.schemas.get(full_name=full_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def validate_catalog_and_schema(self) -> tuple[bool, str]:
        """
        Validates that the specified catalogs and schemas exist for both uc_model_name and evaluation_set_uc_table
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        # Extract catalog and schema from uc_model_name
        model_catalog, model_schema, _ = self.uc_model_name.split(".")

        # Extract catalog and schema from evaluation_set_uc_table
        eval_catalog, eval_schema, _ = self.evaluation_set_uc_table.split(".")

        # Check model catalog and schema
        if not self.check_if_catalog_exists(model_catalog):
            return (
                False,
                f"Model catalog '{model_catalog}' does not exist. Please create it first.",
            )

        if not self.check_if_schema_exists(model_catalog, model_schema):
            return (
                False,
                f"Model schema '{model_schema}' does not exist in catalog '{model_catalog}'. Please create it first.",
            )

        # Check evaluation table catalog and schema
        if not self.check_if_catalog_exists(eval_catalog):
            return (
                False,
                f"Evaluation catalog '{eval_catalog}' does not exist. Please create it first.",
            )

        if not self.check_if_schema_exists(eval_catalog, eval_schema):
            return (
                False,
                f"Evaluation schema '{eval_schema}' does not exist in catalog '{eval_catalog}'. Please create it first.",
            )

        msg = f"All catalogs and schemas exist for both model `{self.uc_model_name}` and evaluation table `{self.evaluation_set_uc_table}`."
        print(msg)
        return (True, msg)
