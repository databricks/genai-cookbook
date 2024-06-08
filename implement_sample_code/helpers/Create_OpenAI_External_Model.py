# Databricks notebook source
# MAGIC %md
# MAGIC # Create (Azure) OpenAI as an [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html)
# MAGIC
# MAGIC External models are third-party models hosted outside of Databricks. Supported by Model Serving, external models allow you to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization. 
# MAGIC
# MAGIC View the [documentation](https://docs.databricks.com/en/generative-ai/external-models/index.html#configure-the-provider-for-an-endpoint) for External Models other than (Azure) OpenAI.

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import os
from databricks.sdk import WorkspaceClient
import mlflow.deployments

# Databricks SDKs
w = WorkspaceClient()
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the key as a Databricks Secret

# COMMAND ----------


# Where to save the secret
SCOPE_NAME = "some_scope_name"
SECRET_NAME = "openai_token"

# OpenAI key
SECRET_TO_SAVE = "your_key_here"

existing_scopes = [scope.name for scope in w.secrets.list_scopes()]
if SCOPE_NAME not in existing_scopes:
    print(f"Creating secret scope `{SCOPE_NAME}`")
    w.secrets.create_scope(scope=SCOPE_NAME)
else:
    print(f"Secret scope `{SCOPE_NAME}` exists")

existing_secrets = [secret.key for secret in w.secrets.list_secrets(scope=SCOPE_NAME)]
if SCOPE_NAME not in existing_scopes:
    print(f"Saving secret to `{SCOPE_NAME}.{SECRET_NAME}`")
    w.secrets.put_secret(scope=SCOPE_NAME, key=SECRET_NAME, string_value=SECRET_TO_SAVE)
else:
    print(f"Secret named `{SCOPE_NAME}.{SECRET_NAME}` already exists - choose a different `SECRET_NAME`")

# COMMAND ----------

# MAGIC %md
# MAGIC # OpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chat models

# COMMAND ----------

# This can be anything, but Databricks suggests naming the endpoint after the model itself e.g., company-gpt-3.5, etc
model_serving_endpoint_name = "name_of_to_be_created_endpoint"

client.create_endpoint(
    name=model_serving_endpoint_name,
    config={
        "served_entities": [
          {
            "name": model_serving_endpoint_name,
            "external_model": {
                "name": "gpt-4-1106-preview", # Name of the OpenAI Model, can be any of gpt-3.5-turbo, gpt-4, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-1106-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview
                "provider": "openai", # openai for Azure OpenAI or OpenAI
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_key": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}", # secret saved above
                },
            },
          }
        ],
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding models

# COMMAND ----------

# This can be anything, but Databricks suggests naming the endpoint after the model itself e.g., company-gpt-3.5, etc
model_serving_endpoint_name = "name_of_to_be_created_endpoint"

client.create_endpoint(
    name=model_serving_endpoint_name,
    config={
        "served_entities": [
          {
            "name": model_serving_endpoint_name,
            "external_model": {
                "name": "text-embedding-3-small", # Name of the OpenAI Model, can be any of text-embedding-ada-002, text-embedding-3-large, text-embedding-3-small
                "task": "llm/v1/embeddings",
                "openai_config": {
                    "openai_api_type": "azure",
                },
            },
          }
        ],
    },
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Azure OpenAI 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chat Models

# COMMAND ----------

# This can be anything, but Databricks suggests naming the endpoint after the model itself e.g., company-gpt-3.5, etc
model_serving_endpoint_name = "name_of_to_be_created_endpoint"

client.create_endpoint(
    name=model_serving_endpoint_name,
    config={
        "served_entities": [
          {
            "name": model_serving_endpoint_name,
            "external_model": {
                "name": "gpt-4-1106-preview", # Name of the OpenAI Model, can be any of gpt-3.5-turbo, gpt-4, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-1106-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview
                "provider": "openai", # openai for Azure OpenAI or OpenAI
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_type": "azure",
                    "openai_api_key": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}", # secret saved above
                    "openai_api_base": "https://my-azure-openai-endpoint.openai.azure.com", #replace with your config
                    "openai_deployment_name": "my-gpt-35-turbo-deployment", #replace with your config
                    "openai_api_version": "2023-05-15" #replace with your config
                },
            },
          }
        ],
    },
)

# COMMAND ----------

# MAGIC %md ## Embedding Models

# COMMAND ----------

# This can be anything, but Databricks suggests naming the endpoint after the model itself e.g., company-gpt-3.5, etc
model_serving_endpoint_name = "name_of_to_be_created_endpoint"

client.create_endpoint(
    name=model_serving_endpoint_name,
    config={
        "served_entities": [
          {
            "name": model_serving_endpoint_name,
            "external_model": {
                "name": "text-embedding-3-small", # Name of the OpenAI Model, can be any of text-embedding-ada-002, text-embedding-3-large, text-embedding-3-small
                "task": "llm/v1/embeddings",
                "openai_config": {
                    "openai_api_type": "azure",
                    "openai_api_key": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}", # secret saved above
                    "openai_api_base": "https://my-azure-openai-endpoint.openai.azure.com", #replace with your config
                    "openai_deployment_name": "my-gpt-35-turbo-deployment", #replace with your config
                    "openai_api_version": "2023-05-15" #replace with your config
                },
            },
          }
        ],
    },
)
