# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-sdk mlflow mlflow-skinny
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config to validate

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState, EndpointType
from mlflow.utils import databricks_utils as du
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
from databricks.sdk.errors import ResourceDoesNotExist
import os

w = WorkspaceClient()
browser_url = du.get_browser_hostname()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if configured locations exist
# MAGIC
# MAGIC If not, creates:
# MAGIC - UC Catalog & Schema
# MAGIC - Vector Search Endpoint
# MAGIC - Folder within UC Volume for streaming checkpoints
# MAGIC
# MAGIC `SOURCE_PATH` will NOT be created, but existance is verified.

# COMMAND ----------

# Check if source location exists
import os

if os.path.isdir(SOURCE_PATH):
    print(f"PASS: `{SOURCE_PATH}` exists")
else:
    print(f"FAIL: `{SOURCE_PATH}` does NOT exist")
    raise ValueError("Please verify that `{SOURCE_PATH}` is a valid UC Volume")

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
    except PermissionDenied as e:
        print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
        raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

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

# MAGIC %md ## Check for Chain code file

# COMMAND ----------

if os.path.exists(CHAIN_CODE_FILE):
    print(f"PASS: Chain file `{CHAIN_CODE_FILE}` exists in the local directory.")
else:
    print(f"FAIL: Chain file `{CHAIN_CODE_FILE}` does not exist - make sure you have copied the chain file from the examples directory.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check LLM & Embedding endpoint

# COMMAND ----------

def check_endpoint(endpoint_name):
    try:
        llm_endpoint = w.serving_endpoints.get(name=endpoint_name)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            print(f"FAIL: Model serving endpoint {endpoint_name} is not in a READY state.  Please visit the status page to debug: https://{browser_url}/ml/endpoints/{endpoint_name}")
            raise ValueError(f"Model Serving endpoint: {endpoint_name} not ready.")
        else: 
            print(f"PASS: Model serving endpoint {endpoint_name} is online & ready.  Details at: https://{browser_url}/ml/endpoints/{endpoint_name}")
    except ResourceDoesNotExist as e:
        print(f"FAIL: Model serving endpoint {endpoint_name} does not exist.  Please create it at: https://{browser_url}/ml/endpoints/")
        raise ValueError(f"Model Serving endpoint: {endpoint_name} not ready.")

# LLM
check_endpoint(embedding_config['embedding_endpoint_name'])
# Embedding
check_endpoint(rag_chain_config['databricks_resources']['llm_endpoint_name'])
