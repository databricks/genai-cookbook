from cookbook.config import SerializableConfig
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType


class DataPipelineOuputConfig(SerializableConfig):
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

    vector_search_endpoint: str
    parsed_docs_table: str
    chunked_docs_table: str
    vector_index: str

    def __init__(
        self,
        *,
        vector_search_endpoint: str,
        parsed_docs_table: Optional[str] = None,
        chunked_docs_table: Optional[str] = None,
        vector_index: Optional[str] = None,
        uc_catalog_name: Optional[str] = None,
        uc_schema_name: Optional[str] = None,
        base_table_name: Optional[str] = None,
        docs_table_postfix: str = "docs",
        chunked_table_postfix: str = "docs_chunked",
        vector_index_postfix: str = "docs_chunked_index",
        version_suffix: Optional[str] = None,
    ):
        """Initialize a new DataPipelineOuputConfig instance.

        Supports two initialization styles:
        1. Direct table names:
            - parsed_docs_table
            - chunked_docs_table
            - vector_index

        2. Generated table names using:
            - uc_catalog_name
            - uc_schema_name
            - base_table_name
            - [optional] postfixes and version_suffix

        Args:
            vector_search_endpoint (str): Name of the vector search endpoint to use
            parsed_docs_table (str, optional): Direct table name for parsed docs
            chunked_docs_table (str, optional): Direct table name for chunked docs
            vector_index (str, optional): Direct name for vector index
            uc_catalog_name (str, optional): Unity Catalog name where tables will be created
            uc_schema_name (str, optional): Schema name within the catalog
            base_table_name (str, optional): Core name used as prefix for all generated tables
            docs_table_postfix (str, optional): Suffix for parsed documents table. Defaults to "docs"
            chunked_table_postfix (str, optional): Suffix for chunked documents table. Defaults to "docs_chunked"
            vector_index_postfix (str, optional): Suffix for vector index. Defaults to "docs_chunked_index"
            version_suffix (str, optional): Version identifier for multiple pipeline versions
        """
        _validate_not_default(vector_search_endpoint)

        if parsed_docs_table and chunked_docs_table and vector_index:
            # Direct table names provided
            if any([uc_catalog_name, uc_schema_name, base_table_name]):
                raise ValueError(
                    "Cannot provide both direct table names and table name generation parameters"
                )
        elif all([uc_catalog_name, uc_schema_name, base_table_name]):
            # Generate table names
            _validate_not_default(uc_catalog_name)
            _validate_not_default(uc_schema_name)
            _validate_not_default(base_table_name)

            parsed_docs_table = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                docs_table_postfix,
                version_suffix,
            )
            chunked_docs_table = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                chunked_table_postfix,
                version_suffix,
            )
            vector_index = _build_table_name(
                uc_catalog_name,
                uc_schema_name,
                base_table_name,
                vector_index_postfix,
                version_suffix,
                escape=False,
            )
        else:
            raise ValueError(
                "Must provide either all direct table names or all table name generation parameters"
            )

        super().__init__(
            parsed_docs_table=parsed_docs_table,
            chunked_docs_table=chunked_docs_table,
            vector_index=vector_index,
            vector_search_endpoint=vector_search_endpoint,
        )

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

    def validate_catalog_and_schema(self) -> tuple[bool, str]:
        """
        Validates that the specified catalog and schema exist
        Returns:
            tuple[bool, str]: A tuple containing (success, error_message).
            If validation passes, returns (True, success_message). If validation fails, returns (False, error_message).
        """

        # Check catalog and schema for parsed_docs_table
        parsed_docs_catalog = _get_uc_catalog_name(self.parsed_docs_table)
        parsed_docs_schema = _get_uc_schema_name(self.parsed_docs_table)
        if not _check_if_catalog_exists(parsed_docs_catalog):
            msg = f"Catalog '{parsed_docs_catalog}' does not exist for parsed_docs_table. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(parsed_docs_catalog, parsed_docs_schema):
            msg = f"Schema '{parsed_docs_schema}' does not exist in catalog '{parsed_docs_catalog}' for parsed_docs_table. Please create it first."
            return (False, msg)

        # Check catalog and schema for chunked_docs_table
        chunked_docs_catalog = _get_uc_catalog_name(self.chunked_docs_table)
        chunked_docs_schema = _get_uc_schema_name(self.chunked_docs_table)
        if not _check_if_catalog_exists(chunked_docs_catalog):
            msg = f"Catalog '{chunked_docs_catalog}' does not exist for chunked_docs_table. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(chunked_docs_catalog, chunked_docs_schema):
            msg = f"Schema '{chunked_docs_schema}' does not exist in catalog '{chunked_docs_catalog}' for chunked_docs_table. Please create it first."
            return (False, msg)

        # Check catalog and schema for vector_index
        vector_index_catalog = _get_uc_catalog_name(self.vector_index)
        vector_index_schema = _get_uc_schema_name(self.vector_index)
        if not _check_if_catalog_exists(vector_index_catalog):
            msg = f"Catalog '{vector_index_catalog}' does not exist for vector_index. Please create it first."
            return (False, msg)
        if not _check_if_schema_exists(vector_index_catalog, vector_index_schema):
            msg = f"Schema '{vector_index_schema}' does not exist in catalog '{vector_index_catalog}' for vector_index. Please create it first."
            return (False, msg)

        msg = f"All catalogs and schemas exist for parsed_docs_table, chunked_docs_table, and vector_index."
        print(msg)
        return (True, msg)


def _escape_uc_fqn(uc_fqn: str) -> str:
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


def _build_table_name(
    uc_catalog_name: str,
    uc_schema_name: str,
    base_table_name: str,
    postfix: str,
    version_suffix: str = None,
    escape: bool = True,
) -> str:
    """Helper to build consistent table names

    Args:
        postfix: The table name postfix to append
        escape: Whether to escape special characters in the table name. Defaults to True.

    Returns:
        The constructed table name, optionally escaped
    """
    suffix = f"__{version_suffix}" if version_suffix else ""
    raw_name = f"{uc_catalog_name}.{uc_schema_name}.{base_table_name}_{postfix}{suffix}"
    return _escape_uc_fqn(raw_name) if escape else raw_name


def _validate_not_default(value: str) -> str:
    if value == "REPLACE_ME":
        raise ValueError(
            "Please replace the default value 'REPLACE_ME' with your actual configuration"
        )
    return value


def _get_uc_catalog_name(uc_fqn: str) -> str:
    unescaped_uc_fqn = uc_fqn.replace("`", "")
    return unescaped_uc_fqn.split(".")[0]


def _get_uc_schema_name(uc_fqn: str) -> str:
    unescaped_uc_fqn = uc_fqn.replace("`", "")
    return unescaped_uc_fqn.split(".")[1]


def _check_if_catalog_exists(uc_catalog_name) -> bool:
    w = WorkspaceClient()
    try:
        w.catalogs.get(name=uc_catalog_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False


def _check_if_schema_exists(uc_catalog_name, uc_schema_name) -> bool:
    w = WorkspaceClient()
    try:
        full_name = f"{uc_catalog_name}.{uc_schema_name}"
        w.schemas.get(full_name=full_name)
        return True
    except (ResourceDoesNotExist, NotFound):
        return False
