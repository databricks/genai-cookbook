from cookbook.config import SerializableConfig
from cookbook.databricks_utils import get_volume_url


from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.catalog import VolumeType
from pydantic import Field, computed_field, field_validator


class UCVolumeSourceConfig(SerializableConfig):
    """
    Source data configuration for the Unstructured Data Pipeline. You can modify this class to add additional configuration settings.

    Args:
      uc_catalog_name (str):
        Required. Name of the Unity Catalog.
      uc_schema_name (str):
        Required. Name of the Unity Catalog schema.
      uc_volume_name (str):
        Required. Name of the Unity Catalog volume.
    """

    @field_validator("uc_catalog_name", "uc_schema_name", "uc_volume_name")
    def validate_not_default(cls, value: str) -> str:
        if value == "REPLACE_ME":
            raise ValueError(
                "Please replace the default value 'REPLACE_ME' with your actual configuration"
            )
        return value

    uc_catalog_name: str = Field(..., min_length=1)
    uc_schema_name: str = Field(..., min_length=1)
    uc_volume_name: str = Field(..., min_length=1)

    @computed_field()
    def volume_path(self) -> str:
        return f"/Volumes/{self.uc_catalog_name}/{self.uc_schema_name}/{self.uc_volume_name}"

    @computed_field()
    def volume_uc_fqn(self) -> str:
        return f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_volume_name}"

    def check_if_volume_exists(self) -> bool:
        w = WorkspaceClient()
        try:
            # Use the computed field instead of reconstructing the FQN
            w.volumes.read(name=self.volume_uc_fqn)
            return True
        except (ResourceDoesNotExist, NotFound):
            return False

    def create_volume(self):
        try:
            w = WorkspaceClient()
            w.volumes.create(
                catalog_name=self.uc_catalog_name,
                schema_name=self.uc_schema_name,
                name=self.uc_volume_name,
                volume_type=VolumeType.MANAGED,
            )
        except ResourceAlreadyExists:
            pass

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

    def create_or_validate_volume(self) -> tuple[bool, str]:
        """
        Validates that the volume exists and creates it if it doesn't
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

        if not self.check_if_volume_exists():
            print(f"Volume {self.volume_path} does not exist. Creating...")
            try:
                self.create_volume()
            except Exception as e:
                msg = f"Failed to create volume: {str(e)}"
                return (False, msg)
            msg = f"Successfully created volume {self.volume_path}. View here: {get_volume_url(self.volume_uc_fqn)}"
            print(msg)
            return (True, msg)

        msg = f"Volume {self.volume_path} exists.  View here: {get_volume_url(self.volume_uc_fqn)}"
        print(msg)
        return (True, msg)

    def list_files(self) -> list[str]:
        """
        Lists all files in the Unity Catalog volume using dbutils.fs.

        Returns:
            list[str]: A list of file paths in the volume

        Raises:
            Exception: If the volume doesn't exist or there's an error accessing it
        """
        if not self.check_if_volume_exists():
            raise Exception(f"Volume {self.volume_path} does not exist")

        w = WorkspaceClient()
        try:
            # List contents using dbutils.fs
            files = w.dbutils.fs.ls(self.volume_path)
            return [file.name for file in files]
        except Exception as e:
            raise Exception(f"Failed to list files in volume: {str(e)}")
