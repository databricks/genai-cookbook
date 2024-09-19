# Databricks notebook source
# Install these if running this Notebook on its own
# %pip install -U -qqqq databricks-sdk mlflow-skinny
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
        print(
            f"FAIL: {retriever_config.vector_search_index} does not exist.  Please check the vector search index name."
        )

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
            raise ValueError(
                f"Model Serving endpoint: {llm_config.llm_endpoint_name} not ready."
            )
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
        raise ValueError(
            f"Model Serving endpoint: {llm_config.llm_endpoint_name} not ready."
        )

