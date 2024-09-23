# Databricks notebook source
# Install these if running this Notebook on its own
# %pip install -U -qqqq databricks-sdk mlflow mlflow-skinny
# dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from mlflow.utils import databricks_utils as du
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os
from databricks.sdk.service.compute import DataSecurityMode
from pyspark.sql import SparkSession

w = WorkspaceClient()
browser_url = du.get_browser_hostname()

# COMMAND ----------

# Check if source location exists
import os

if os.path.isdir(SOURCE_UC_VOLUME):
    print(f"PASS: `{SOURCE_UC_VOLUME}` exists")
else:
    print(f"`{SOURCE_UC_VOLUME}` does NOT exist, trying to create")

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import catalog
    from databricks.sdk.errors import ResourceAlreadyExists

    w = WorkspaceClient()

    volume_name = SOURCE_UC_VOLUME[9:].split("/")[2]
    uc_catalog = SOURCE_UC_VOLUME[9:].split("/")[0]
    uc_schema = SOURCE_UC_VOLUME[9:].split("/")[1]
    try:
        created_volume = w.volumes.create(
            catalog_name=uc_catalog,
            schema_name=uc_schema,
            name=volume_name,
            volume_type=catalog.VolumeType.MANAGED,
        )
        print(f"PASS: Created `{SOURCE_UC_VOLUME}`")
    except Exception as e:
        print(f"`FAIL: {SOURCE_UC_VOLUME}` does NOT exist, could not create due to {e}")
        raise ValueError("Please verify that `{SOURCE_UC_VOLUME}` is a valid UC Volume")

# COMMAND ----------

# Create the Vector Search endpoint if it does not exist
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType

w = WorkspaceClient()
vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
    print(
        f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes..."
    )
    w.vector_search_endpoints.create_endpoint_and_wait(
        VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD
    )

# Make sure vector search endpoint is online and ready.
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
    VECTOR_SEARCH_ENDPOINT
)

print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Embedding endpoint

# COMMAND ----------

# TODO: Add check for tool calling support


def validate_embedding_endpoint(endpoint_name, task_type):
    try:
        llm_endpoint = w.serving_endpoints.get(name=endpoint_name)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            print(
                f"FAIL: Model serving endpoint {endpoint_name} is not in a READY state.  Please visit the status page to debug: https://{browser_url}/ml/endpoints/{endpoint_name}"
            )
            raise ValueError(f"Model Serving endpoint: {endpoint_name} not ready.")
        else:
            if llm_endpoint.task != task_type:
                print(
                    f"FAIL: Model serving endpoint {endpoint_name} is online & ready, but does not support task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{endpoint_name}"
                )
            else:
                print(
                    f"PASS: Model serving endpoint {endpoint_name} is online & ready and supports task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{endpoint_name}"
                )
    except ResourceDoesNotExist as e:
        print(
            f"FAIL: Model serving endpoint {endpoint_name} does not exist.  Please create it at: https://{browser_url}/ml/endpoints/"
        )
        raise ValueError(f"Model Serving endpoint: {endpoint_name} not ready.")


# Embedding
validate_embedding_endpoint(EMBEDDING_MODEL_ENDPOINT, "llm/v1/embeddings")

# COMMAND ----------

# Check if source location exists
import os


def create_or_check_volume_path(volume_path: str) -> None:
    if os.path.isdir(volume_path):
        print(f"PASS: `{volume_path}` exists")
    else:
        print(f"`{volume_path}` does NOT exist, trying to create")

        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service import catalog
        from databricks.sdk.errors import ResourceAlreadyExists

        w = WorkspaceClient()

        volume_parts = volume_path.removeprefix("dbfs:/Volumes")
        uc_catalog, uc_schema, volume_name, volume_path = volume_parts.split("/", 4)
        try:
            created_volume = w.volumes.create(
                catalog_name=uc_catalog,
                schema_name=uc_schema,
                name=volume_name,
                volume_type=catalog.VolumeType.MANAGED,
            )
            print(f"PASS: Created `{volume_path}`")
        except Exception as e:
            print(f"`FAIL: {volume_path}` does NOT exist, could not create due to {e}")
            raise ValueError("Please verify that `{volume_path}` is a valid UC Volume")

# COMMAND ----------

# Create the Vector Search endpoint if it does not exist
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType


def create_or_check_vector_search_endpoint(vector_search_endpoint: str):
    w = WorkspaceClient()
    vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
    if sum([vector_search_endpoint == ve.name for ve in vector_search_endpoints]) == 0:
        print(
            f"Please wait, creating Vector Search endpoint `{vector_search_endpoint}`.  This can take up to 20 minutes..."
        )
        w.vector_search_endpoint.create_endpoint_and_wait(
            vector_search_endpoint, endpoint_type=EndpointType.STANDARD
        )

    # Make sure vector search endpoint is online and ready.
    w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
        vector_search_endpoint
    )

    print(f"PASS: Vector Search endpoint `{vector_search_endpoint}` exists")
