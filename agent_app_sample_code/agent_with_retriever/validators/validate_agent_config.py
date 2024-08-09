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


# # Get the cluster ID
# spark_session = SparkSession.getActiveSession()
# cluster_id = spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", None)

# # # Get the current cluster name
# # try:
# #   cluster_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("clusterId").get()
# # except Exception as e:

# cluster_info = w.clusters.get(cluster_id)

# # Check if a single user cluster
# # Serverless will return None here
# if not cluster_info.data_security_mode == DataSecurityMode.SINGLE_USER:
#   raise ValueError(f"FAIL: Current cluster is not a Single User cluster.  This notebooks currently require a single user cluster.  Please create a single user cluster: https://docs.databricks.com/en/compute/configure.html#single-node-or-multi-node-compute")

# # Check for 14.3+
# major_version = int(cluster_info.spark_version.split(".")[0])
# minor_version = int(cluster_info.spark_version.split(".")[1])

# if not ((major_version==15) or (major_version==14 and minor_version>=3)):
#   raise ValueError(f"FAIL: Current cluster version {major_version}.{minor_version} is less than DBR or MLR 14.3.  Please create a DBR 14.3+ single user cluster.")
# else:
#   print("PASS: Running on a single user cluster version with DBR or MLR 14.3+ ({major_version}.{minor_version}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if configured locations exist
# MAGIC
# MAGIC If not, creates:
# MAGIC - UC Catalog & Schema

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.errors import NotFound, PermissionDenied
# w = WorkspaceClient()

# # Create UC Catalog if it does not exist, otherwise, raise an exception
# try:
#     _ = w.catalogs.get(UC_CATALOG)
#     print(f"PASS: UC catalog `{UC_CATALOG}` exists")
# except NotFound as e:
#     print(f"`{UC_CATALOG}` does not exist, trying to create...")
#     try:
#         _ = w.catalogs.create(name=UC_CATALOG)
#     except PermissionDenied as e:
#         print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
#         raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
        
# # Create UC Schema if it does not exist, otherwise, raise an exception
# try:
#     _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
#     print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
# except NotFound as e:
#     print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
#     try:
#         _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
#         print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
#     except PermissionDenied as e:
#         print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
#         raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check if Vector Search Index exists

# COMMAND ----------

# Create the Vector Search endpoint if it does not exist
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
from databricks.sdk.errors import ResourceDoesNotExist
w = WorkspaceClient()

def validate_retriever_config(retriever_config: AgentConfig):
  try:
    w.vector_search_indexes.get_index(retriever_config.vector_search_index)
    print(f"PASS: {retriever_config.vector_search_index} exists.")
  except ResourceDoesNotExist as e:
    print(f"FAIL: {retriever_config.vector_search_index} does not exist.  Please check the vector search index name.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check LLM endpoint

# COMMAND ----------

# TODO: Add check for tool calling support

def validate_llm_config(llm_config: LLMConfig):
    task_type = "llm/v1/chat"
    try:
        llm_endpoint = w.serving_endpoints.get(name=llm_config.llm_endpoint_name)
        if llm_endpoint.state.ready != EndpointStateReady.READY:
            print(
                f"FAIL: Model serving endpoint {llm_config.llm_endpoint_name} is not in a READY state.  Please visit the status page to debug: https://{browser_url}/ml/endpoints/{llm_config.llm_endpoint_name}"
            )
            raise ValueError(f"Model Serving endpoint: {llm_config.llm_endpoint_name} not ready.")
        else:
            if llm_endpoint.task != task_type:
                print(
                    f"FAIL: Model serving endpoint {llm_config.llm_endpoint_name} is online & ready, but does not support task type /{task_type}.  Details at: https://{browser_url}/ml/endpoints/{llm_config.llm_endpoint_name}"
                )
            else:
                print(
                    f"PASS: Model serving endpoint {llm_config.llm_endpoint_name} is online & ready and supports task type /{task_type}.  Details at: https://{browser_url}/ml/endpoints/{llm_config.llm_endpoint_name}"
                )
    except ResourceDoesNotExist as e:
        print(
            f"FAIL: Model serving endpoint {llm_config.llm_endpoint_name} does not exist.  Please create it at: https://{browser_url}/ml/endpoints/"
        )
        raise ValueError(f"Model Serving endpoint: {llm_config.llm_endpoint_name} not ready.")

# Embedding
#check_endpoint(llm_config.llm_endpoint_name, "llm/v1/chat")
