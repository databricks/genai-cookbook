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

# MAGIC %md
# MAGIC ## Check if running on Single User 14.3+ cluster

# COMMAND ----------


# Get the cluster ID
spark_session = SparkSession.getActiveSession()
cluster_id = spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", None)

# # Get the current cluster name
# try:
#   cluster_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()
# except Exception as e:

cluster_info = w.clusters.get(cluster_id)

# Check if a single user cluster
# Serverless will return None here
# if not cluster_info.data_security_mode == DataSecurityMode.SINGLE_USER:
#   raise ValueError(f"FAIL: Current cluster is not a Single User cluster.  This notebooks currently require a single user cluster.  Please create a single user cluster: https://docs.databricks.com/en/compute/configure.html#single-node-or-multi-node-compute")

# Check for 14.3+
major_version = int(cluster_info.spark_version.split(".")[0])
minor_version = int(cluster_info.spark_version.split(".")[1])

if not ((major_version==15) or (major_version==14 and minor_version>=3)):
  raise ValueError(f"FAIL: Current cluster version {major_version}.{minor_version} is less than DBR or MLR 14.3.  Please create a DBR 14.3+ single user cluster.")
else:
  print("PASS: Running on a single user cluster version with DBR or MLR 14.3+ ({major_version}.{minor_version}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if configured locations exist
# MAGIC
# MAGIC If not, creates:
# MAGIC - UC Catalog & Schema
# MAGIC - Vector Search Endpoint
# MAGIC - Source Path

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

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

    volume_name = SOURCE_UC_VOLUME[9:].split('/')[2]
    uc_catalog = SOURCE_UC_VOLUME[9:].split('/')[0]
    uc_schema = SOURCE_UC_VOLUME[9:].split('/')[1]
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
    print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes...")
    w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

# Make sure vector search endpoint is online and ready.
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)

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
          f"FAIL: Model serving endpoint {endpoint_name} is online & ready, but does not support task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{endpoint_name}")
      else:
        print(
          f"PASS: Model serving endpoint {endpoint_name} is online & ready and supports task type {task_type}.  Details at: https://{browser_url}/ml/endpoints/{endpoint_name}")
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

    volume_parts = volume_path.removeprefix('dbfs:/Volumes')
    uc_catalog, uc_schema, volume_name, volume_path = volume_parts.split('/', 4)
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
      print(f"Please wait, creating Vector Search endpoint `{vector_search_endpoint}`.  This can take up to 20 minutes...")
      w.vector_search_endpoint.create_endpoint_and_wait(vector_search_endpoint, endpoint_type=EndpointType.STANDARD)

  # Make sure vector search endpoint is online and ready.
  w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(vector_search_endpoint)

  print(f"PASS: Vector Search endpoint `{vector_search_endpoint}` exists")

# COMMAND ----------

# def create_or_check_vector_search_endpoint
