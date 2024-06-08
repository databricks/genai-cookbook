# Databricks notebook source
# MAGIC %pip install -U -qqqq mlflow mlflow-skinny

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application configuration
# MAGIC
# MAGIC To begin with, we simply need to configure the following:
# MAGIC 1. `RAG_APP_NAME`: The name of the RAG application.  Used to name the app's Unity Catalog model and is prepended to the output Delta Tables + Vector Indexes
# MAGIC 2. `UC_CATALOG` & `UC_SCHEMA`: [Create a Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored
# MAGIC 3. `UC_MODEL_NAME`: Unity Catalog location to log and store the chain's model
# MAGIC 4. `VECTOR_SEARCH_ENDPOINT`: [Create a Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to host the resulting vector index
# MAGIC 5. `SOURCE_PATH`: A [UC Volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html#create-and-work-with-volumes) that contains the source documents for your application.
# MAGIC 6. `MLFLOW_EXPERIMENT_NAME`: MLflow Experiment to track all experiments for this application.  Using the same experiment allows you to track runs across Notebooks and have unified lineage and governance for your application.
# MAGIC 7. `EVALUATION_SET_FQN`: Delta Table where your evaluation set will be stored.  In the POC, we will seed the evaluation set with feedback you collect from your stakeholders.
# MAGIC
# MAGIC After finalizing your configuration, optionally run `01_validate_config` to check that all locations exist. 

# COMMAND ----------

# By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "")

# COMMAND ----------

# The name of the RAG application.  This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes
RAG_APP_NAME = 'my_agent_app'

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'{user_name}_catalog'
UC_SCHEMA = f'rag_{user_name}'

## UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{user_name}_vector_search'

# Source location for documents
# You need to create this location and add files
SOURCE_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/source_docs"

############################
##### We suggest accepting these defaults unless you need to change them. ######
############################

EVALUATION_SET_FQN = f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}_evaluation_set"

# MLflow experiment name
# Using the same MLflow experiment for a single app allows you to compare runs across Notebooks
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{RAG_APP_NAME}"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# MLflow Run Names
# These Runs will store your initial POC application.  They are later used to evaluate the POC model against your experiments to improve quality.

# Data pipeline MLflow run name
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
# Chain MLflow run name
POC_CHAIN_RUN_NAME = "poc"

# COMMAND ----------

print(f"RAG_APP_NAME {RAG_APP_NAME}")
print(f"UC_CATALOG {UC_CATALOG}")
print(f"UC_SCHEMA {UC_SCHEMA}")
print(f"UC_MODEL_NAME {UC_MODEL_NAME}")
print(f"VECTOR_SEARCH_ENDPOINT {VECTOR_SEARCH_ENDPOINT}")
print(f"SOURCE_PATH {SOURCE_PATH}")
print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")
