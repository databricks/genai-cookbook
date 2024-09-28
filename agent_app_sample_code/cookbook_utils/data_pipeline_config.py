from dataclasses import dataclass, field, asdict
import yaml
import json
from .cookbook_dataclass import CookbookConfig
from databricks.sdk import WorkspaceClient
import os
from databricks.sdk.service.catalog import VolumeType
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady

from mlflow.utils import databricks_utils as du

from .recursive_character_text_splitter import (
    EMBEDDING_MODELS,
    detect_fmapi_embedding_model_type,
    validate_chunk_size,
)

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn):
    table_fqdn = table_fqdn.replace("`", "")
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

@dataclass
class UnstructuredDataPipelineSourceConfig(CookbookConfig):
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

    uc_catalog_name: str
    uc_schema_name: str
    uc_volume_name: str

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.uc_catalog_name}/{self.uc_schema_name}/{self.uc_volume_name}"

    def check_if_volume_exists(self) -> bool:
        if os.path.isdir(self.volume_path):
            print(f"\nPASS: `{self.volume_path}` exists")
            return True
        else:
            print(f"\n`{self.volume_path}` does not exist")
            return False

    def create_volume(self) -> bool:
        try:
            w = WorkspaceClient()
            created_volume = w.volumes.create(
                catalog_name=self.uc_catalog_name,
                schema_name=self.uc_schema_name,
                name=self.uc_volume_name,
                volume_type=VolumeType.MANAGED,
            )
            print(f"\nPASS: Created `{self.volume_path}`")
            return True
        except ResourceAlreadyExists as e:
            print(
                f"\nCould not create volume `{self.volume_path}` since it already exists"
            )
            return True
        except Exception as e:
            print(
                f'\nFAIL: could not create {self.volume_path}` \n\nError details: "{e}"\n\nPlease retry with a `uc_volume_name` that you have permissions to access OR provide a `uc_catalog_name` & `uc_schema_name` where you have CREATE VOLUME permission.'
            )
            return False

    def create_or_check_volume(self) -> bool:
        if not self.check_if_volume_exists():
            return self.create_volume()
        else:
            return True


@dataclass
class ChunkingConfig(CookbookConfig):
    """
    Configuration for the Unstructured Data Pipeline.

    Args:
        embedding_model_endpoint (str):
            Embedding model endpoint hosted on Model Serving.  Default is `databricks-gte-large`.  This can be an External Model, such as OpenAI or a Databricks hosted model on Foundational Model API. The list of Databricks hosted models can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        chunk_size_tokens (int):
            The size of each chunk of the document in tokens. Default is 2048.
        chunk_overlap_tokens (int):
            The overlap of tokens between chunks. Default is 512.
    """

    embedding_model_endpoint: str = "databricks-gte-large-en"
    chunk_size_tokens: int = 2048
    chunk_overlap_tokens: int = 256

    def validate_embedding_endpoint(self) -> bool:
        task_type = "llm/v1/embeddings"
        w = WorkspaceClient()
        try:
            browser_url = du.get_browser_hostname()
            llm_endpoint = w.serving_endpoints.get(name=self.embedding_model_endpoint)
            if llm_endpoint.state.ready != EndpointStateReady.READY:
                print(
                    f"\nFAIL: Model serving endpoint {self.embedding_model_endpoint} is not in a READY state.  Please visit the status page to debug: https://{browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
                )
                return False
            else:
                if llm_endpoint.task != task_type:
                    print(
                        f"\nFAIL: Model serving endpoint {self.embedding_model_endpoint} is online & ready, but does not support task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
                    )
                    return False
                else:
                    print(
                        f"\nPASS: Model serving endpoint {self.embedding_model_endpoint} is online & ready and supports task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{self.embedding_model_endpoint}"
                    )
                    return True
        except ResourceDoesNotExist as e:
            print(
                f"\nFAIL: Model serving endpoint {self.embedding_model_endpoint} does not exist.  Please create it at: https://{browser_url}/ml/endpoints/"
            )
            return False

    def validate_chunk_size_and_overlap(self) -> bool:
        embedding_model_name, chunk_spec = detect_fmapi_embedding_model_type(
            self.embedding_model_endpoint
        )
        if chunk_spec is None or embedding_model_name is None:
            print(
                f"\nFAIL: {self.embedding_model_endpoint} is not currently supported by the default chunking logic.  Please update `cookbook_utils/recursive_character_text_splitter.py` to add support."
            )
            return False

        chunk_spec["chunk_size_tokens"] = self.chunk_size_tokens
        chunk_spec["chunk_overlap_tokens"] = self.chunk_overlap_tokens

        is_valid, reason = validate_chunk_size(chunk_spec)
        print("\n" + reason)
        return is_valid


@dataclass
class UnstructuredDataPipelineStorageConfig(CookbookConfig):
    """
    Storage configuration for the Unstructured Data Pipeline.

    Args:
      uc_catalog (str):
       Required.  Unity Catalog catalog name.

      uc_schema (str):
        Required. Unity Catalog schema name.

      uc_asset_prefix (str):
        Required if using default asset names. Prefix for the UC objects that will be created within the schema.  Typically a short name to identify the Agent e.g., "my_agent_app", used to generate the default names for `source_uc_volume`, `parsed_docs_table`, `chunked_docs_table`, and `vector_index_name`.

      vector_search_endpoint (str):
        Required. Vector Search endpoint where index is loaded.

      parsed_docs_table (str):
        Optional. UC location of the Delta Table to store parsed documents. Default is {uc_asset_prefix}_docs`

      chunked_docs_table (str):
        Optional. UC location of the Delta Table to store chunks of the parsed documents. Default is `{uc_asset_prefix}_docs_chunked`

      vector_index_name (str):
        Optional. UC location of the Vector Search index that is created from `chunked_docs_table`. Default is `{uc_asset_prefix}_docs_chunked_index`

      tag (str):
        Optional. A tag to append to the asset names.  Use to differentiate between versions when iterating on chunking/parsing/embedding configs.  If provided and does not start with "__", it will be prefixed with "__".
    """

    uc_catalog_name: str
    uc_schema_name: str
    vector_search_endpoint: str
    tag: str = None
    uc_asset_prefix: str = field(default=None)
    parsed_docs_table: str = field(default=None)
    chunked_docs_table: str = field(default=None)
    vector_index: str = field(default=None)

    def __post_init__(self):
        """
        Post-initialization to set default values for fields that are not provided by the user.
        """
        if self.are_any_uc_asset_names_empty() and self.uc_asset_prefix is None:
            raise ValueError(
                "Must provide `uc_asset_prefix` since you did not provide a value for 1+ of `parsed_docs_table`, `chunked_docs_table`, or `vector_index`.  `uc_asset_prefix` is used to compute the default values for these properties."
            )

        # add "_" to the tag if it doesn't exist
        table_postfix = "" if self.tag is None else self.tag
        table_postfix = (
            f"__{table_postfix}"
            if (table_postfix[:2] != "__" and len(table_postfix) > 0)
            else table_postfix
        )

        if self.parsed_docs_table is None:
            self.parsed_docs_table = self.get_uc_fqn(f"docs{table_postfix}")
        else:
            self.parsed_docs_table = self.get_uc_fqn_for_asset_name(
                self.parsed_docs_table + table_postfix
            )

        if self.chunked_docs_table is None:
            self.chunked_docs_table = self.get_uc_fqn(f"docs_chunked{table_postfix}")
        else:
            self.chunked_docs_table = self.get_uc_fqn_for_asset_name(
                self.chunked_docs_table + table_postfix
            )

        if self.vector_index is None:
            self.vector_index = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_asset_prefix}_docs_chunked_index{table_postfix}"
        else:
            self.vector_index = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.vector_index}{table_postfix}"

    def are_any_uc_asset_names_empty(self) -> bool:
        """
        Check if any of the Unity Catalog asset names are empty.

        Returns:
            bool: True if any of the asset names (`parsed_docs_table`, `chunked_docs_table`, `vector_index`) are None, otherwise False.
        """
        if (
            self.parsed_docs_table is None
            or self.chunked_docs_table is None
            or self.vector_index is None
        ):
            return True
        else:
            return False

    def get_uc_fqn(self, asset_name: str) -> str:
        """
        Generate the fully qualified name (FQN) for a Unity Catalog asset.

        Args:
            asset_name (str): The name of the asset to generate the FQN for.

        Returns:
            str: The fully qualified name of the asset, with necessary escaping for special characters.
        """
        uc_fqn = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_asset_prefix}_{asset_name}"

        return self.escape_uc_fqn(uc_fqn)

    def get_uc_fqn_for_asset_name(self, asset_name):
        uc_fqn = f"{self.uc_catalog_name}.{self.uc_schema_name}.{asset_name}"

        return self.escape_uc_fqn(uc_fqn)

    def escape_uc_fqn(self, uc_fqn):
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
            print(
                f"\nVector Search endpoint `{self.vector_search_endpoint}` does not exist"
            )
            return False
        else:
            print(
                f"\nPASS: Vector Search endpoint `{self.vector_search_endpoint}` exists"
            )
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
        print(
            f"PASS: Vector Search endpoint `{self.vector_search_endpoint}` created and ready."
        )
        return True
    
    def create_or_check_vector_search_endpoint(self) -> bool:
        if not self.check_if_vector_search_endpoint_exists():
            return self.create_vector_search_endpoint()
        else:
            return True


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType