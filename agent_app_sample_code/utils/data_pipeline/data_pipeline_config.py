from pydantic import BaseModel, Field, computed_field
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType
from databricks.sdk.errors import NotFound
from utils.cookbook.databricks_utils import get_volume_url
import yaml
from utils.data_pipeline.recursive_character_text_splitter import ChunkingConfig


class UCVolumeSourceConfig(BaseModel):
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

    uc_catalog_name: str = Field(..., min_length=1)
    uc_schema_name: str = Field(..., min_length=1)
    uc_volume_name: str = Field(..., min_length=1)

    @computed_field
    def volume_path(self) -> str:
        return f"/Volumes/{self.uc_catalog_name}/{self.uc_schema_name}/{self.uc_volume_name}"

    @computed_field
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


class DataPipelineOuputConfig(BaseModel):
    """Configuration for managing output locations and naming conventions in the data pipeline.

    This class handles the configuration of table names and vector search endpoints for the data pipeline.
    It follows a consistent naming pattern for all generated tables and provides version control capabilities.

    Naming Convention:
        {catalog}.{schema}.{base_table_name}_{table_postfix}__{version_suffix}

    Generated Tables:
        1. Parsed docs table: Stores the raw parsed documents
        2. Chunked docs table: Stores the documents split into chunks
        3. Vector index: Stores the vector embeddings for search

    Args:
        uc_catalog_name (str): Unity Catalog name where tables will be created
        uc_schema_name (str): Schema name within the catalog
        base_table_name (str): Core name used as prefix for all generated tables
        docs_table_postfix (str, optional): Suffix for the parsed documents table. Defaults to "docs"
        chunked_table_postfix (str, optional): Suffix for the chunked documents table. Defaults to "docs_chunked"
        vector_index_postfix (str, optional): Suffix for the vector index. Defaults to "docs_chunked_index"
        version_suffix (str, optional): Version identifier (e.g., 'v1', 'test') to maintain multiple pipeline versions
        vector_search_endpoint (str): Name of the vector search endpoint to use

    Examples:
        With version_suffix="v1":
            >>> config = DataPipelineOuputConfig(
            ...     uc_catalog_name="my_catalog",
            ...     uc_schema_name="my_schema",
            ...     base_table_name="agent",
            ...     version_suffix="v1"
            ... )
            # Generated tables:
            # - my_catalog.my_schema.agent_docs__v1
            # - my_catalog.my_schema.agent_docs_chunked__v1
            # - my_catalog.my_schema.agent_docs_chunked_index__v1

        Without version_suffix:
            # - my_catalog.my_schema.agent_docs
            # - my_catalog.my_schema.agent_docs_chunked
            # - my_catalog.my_schema.agent_docs_chunked_index
    """

    uc_catalog_name: str = Field(..., min_length=1)
    uc_schema_name: str = Field(..., min_length=1)

    base_table_name: str = Field(..., min_length=1)  # e.g. "agent"

    docs_table_postfix: str = "docs"
    chunked_table_postfix: str = "docs_chunked"
    vector_index_postfix: str = "docs_chunked_index"

    version_suffix: Optional[str] = Field(
        default=None,
        description="Optional version identifier (e.g. 'v1', 'test') that will be appended to all table names. "
        "Use this to maintain multiple versions of the pipeline output with the same base_table_name.",
    )

    vector_search_endpoint: str

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

    def _build_table_name(self, postfix: str, escape: bool = True) -> str:
        """Helper to build consistent table names

        Args:
            postfix: The table name postfix to append
            escape: Whether to escape special characters in the table name. Defaults to True.

        Returns:
            The constructed table name, optionally escaped
        """
        suffix = f"__{self.version_suffix}" if self.version_suffix else ""
        raw_name = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.base_table_name}_{postfix}{suffix}"
        return self.escape_uc_fqn(raw_name) if escape else raw_name

    @property
    def parsed_docs_table(self) -> str:
        """Returns fully qualified name for parsed docs table"""
        return self._build_table_name(self.docs_table_postfix)

    @property
    def chunked_docs_table(self) -> str:
        """Returns fully qualified name for chunked docs table"""
        return self._build_table_name(self.chunked_table_postfix)

    @property
    def vector_index(self) -> str:
        """Returns fully qualified name for vector index"""
        return self._build_table_name(self.vector_index_postfix, escape=False)

    def check_if_vector_search_endpoint_exists(self):
        w = WorkspaceClient()
        vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
        if (
            sum(
                [
                    self.vector_search_endpoint == ve.name
                    for ve in vector_search_endpoints
                ]
            )
            == 0
        ):
            return False
        else:
            return True

    def create_vector_search_endpoint(self):
        w = WorkspaceClient()
        print(
            f"Please wait, creating Vector Search endpoint `{self.vector_search_endpoint}`.  This can take up to 20 minutes..."
        )
        w.vector_search_endpoints.create_endpoint_and_wait(
            self.vector_search_endpoint, endpoint_type=EndpointType.STANDARD
        )
        # Make sure vector search endpoint is online and ready.
        w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            self.vector_search_endpoint
        )

    def create_or_validate_vector_search_endpoint(self):
        if not self.check_if_vector_search_endpoint_exists():
            self.create_vector_search_endpoint()
        return self.validate_vector_search_endpoint()

    def validate_vector_search_endpoint(self) -> tuple[bool, str]:
        """
        Validates that the specified Vector Search endpoint exists
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """
        if not self.check_if_vector_search_endpoint_exists():
            msg = f"Vector Search endpoint '{self.vector_search_endpoint}' does not exist. Please either manually create it or call `output_config.create_or_validate_vector_search_endpoint()` to create it."
            return (False, msg)

        msg = f"Vector Search endpoint '{self.vector_search_endpoint}' exists."
        print(msg)
        return (True, msg)

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


class DataPipelineConfig(BaseModel):
    source: UCVolumeSourceConfig
    output: DataPipelineOuputConfig
    chunking_config: ChunkingConfig

    def to_yaml(self) -> str:
        # exclude_none = True prevents unused parameters from being included in the config
        data = self.dict(exclude_none=True)
        return yaml.dump(data, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataPipelineConfig":
        # Load the data from YAML
        config_dict = yaml.safe_load(yaml_str)
        return cls(**config_dict)
