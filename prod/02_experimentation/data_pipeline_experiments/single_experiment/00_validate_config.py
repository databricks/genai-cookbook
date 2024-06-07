# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

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
        raise ValueError("Unity Catalog `{UC_CATALOG}` does not exist.")
        
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

print(f"Using Vector Search endpoint: `{VECTOR_SEARCH_ENDPOINT}`")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the `checkpoints` Volume if it does not exist

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

volume_name = CHECKPOINTS_VOLUME_PATH.split('/')[-1]
try:
    created_volume = w.volumes.create(
        catalog_name=UC_CATALOG,
        schema_name=UC_SCHEMA,
        name=volume_name,
        # storage_location=external_location.url,
        volume_type=catalog.VolumeType.MANAGED,
    )
    print(f"Created /Volumes/{UC_CATALOG}/{UC_SCHEMA}/{volume_name}/")
except ResourceAlreadyExists as e:
    print(f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{volume_name}/ exists")
