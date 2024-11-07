from pydantic import (
    BaseModel,
    Field,
    root_validator,
    computed_field,
    field_validator,
    FieldValidationInfo,
)
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    ResourceAlreadyExists,
    ResourceDoesNotExist,
    NotFound,
    PermissionDenied,
)


from pydantic import BaseModel, Field, computed_field
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType
from databricks.sdk.errors import NotFound
from cookbook.databricks_utils import get_volume_url
import yaml
from cookbook.config.data_pipeline.recursive_text_splitter import (
    RecursiveTextSplitterChunkingConfig,
)


class AgentStorageConfig(BaseModel):
    """
    Source data configuration for the Unstructured Data Pipeline. You can modify this class to add additional configuration settings.

    Args:
      uc_catalog_name (str):
        Required. Name of the Unity Catalog.
      uc_schema_name (str):
        Required. Name of the Unity Catalog schema.
      uc_model_name (str):
        Required. Name of the Unity Catalog model.
    """

    uc_catalog_name: str = Field(..., min_length=1)
    uc_schema_name: str = Field(..., min_length=1)
    uc_model_name: str = Field(..., min_length=1)
    evaluation_set_uc_table: str = Field(..., min_length=1)
    mlflow_experiment_name: str = Field(None)

    @computed_field
    def model_uc_fqn(self) -> str:
        return f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_model_name}"

    @computed_field
    def evaluation_set_uc_fqn(self) -> str:
        fqn = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.evaluation_set_uc_table}"
        return self.escape_uc_fqn(fqn)

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

    def check_if_catalog_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            w.catalogs.get(name=self.uc_catalog_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def check_if_schema_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            full_name = f"{self.uc_catalog_name}.{self.uc_schema_name}"
            w.schemas.get(full_name=full_name)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def validate_catalog_and_schema(self) -> tuple[bool, str]:
        """
        Validates that the specified catalog and schema exist
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        if not self.check_if_catalog_exists():
            msg = f"Catalog '{self.uc_catalog_name}' does not exist. Please create it first."
            return (False, msg)

        if not self.check_if_schema_exists():
            msg = f"Schema '{self.uc_schema_name}' does not exist in catalog '{self.uc_catalog_name}'. Please create it first."
            return (False, msg)

        msg = f"Catalog '{self.uc_catalog_name}' and schema '{self.uc_schema_name}' exist."
        print(msg)
        return (True, msg)

    def to_yaml(self) -> str:
        # exclude_none = True prevents unused parameters from being included in the config
        data = self.dict(exclude_none=True)
        return yaml.dump(data, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AgentStorageConfig":
        # Load the data from YAML
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)
