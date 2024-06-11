# Databricks notebook source
# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install -U transformers torch mlflow sentence_transformers einops

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointStateReady
import time
from huggingface_hub import snapshot_download
from mlflow.utils import databricks_utils as du

# COMMAND ----------

# MAGIC %md
# MAGIC ## What model to load?

# COMMAND ----------

dbutils.widgets.dropdown(
    name='model_name',
    defaultValue='Alibaba-NLP/gte-large-en-v1.5',
    choices=[
        'Alibaba-NLP/gte-large-en-v1.5',
        'nomic-ai/nomic-embed-text-v1',
        'intfloat/e5-large-v2'
    ],
    label='Hugging Face Model Name (must support Sentence Transformers)'
)

# Retrieve the values from the widgets
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md ## Model Serving config

# COMMAND ----------

# GPU Model Serving configuration
# https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu-workload-types
serving_workload_type = "GPU_MEDIUM"
serving_workload_size = "Small"
serving_scale_to_zero_enabled = "False"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other config

# COMMAND ----------

mlflow_artifact_path = "model"
example_inputs = ["This is an example sentence", "Each sentence is converted"]

# Remove model_provider from `model_provider/model_name`
model_stub_name = model_name.split("/")[1]

# Use Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Configure Databricks clients
client = mlflow.MlflowClient()
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model name & UC location

# COMMAND ----------

# Create widgets for user input UC
dbutils.widgets.text("uc_catalog", "", "Unity Catalog")
dbutils.widgets.text("uc_schema", "", "Unity Catalog Schema")

# Retrieve the values from the widgets
uc_catalog = dbutils.widgets.get("uc_catalog")
uc_schema = dbutils.widgets.get("uc_schema")

if uc_catalog == "" or uc_schema == "":
  raise ValueError("Please set UC Catalog & Schema to continue.")

# MLflow model name: The Model Registry will use this name for the model.
registered_model_name = f'{uc_catalog}.{uc_schema}.{model_stub_name.replace(".", "_")}'
# Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

endpoint_name = f'{registered_model_name.replace(".", "_")}'

# Workspace URL for REST API call
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)

# Get current user's token for API call.
# It is better to create endpoints using a token created for a Service Principal so that the endpoint can outlive a user's tenure at the company.
# See https://docs.databricks.com/dev-tools/service-principals.html
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the model

# COMMAND ----------

# If the model has been downloaded in previous cells, this will not repetitively download large model files, but only the remaining files in the repo
snapshot_location = snapshot_download(repo_id=model_name,  cache_dir="/local_disk0/embedding-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the model locally

# COMMAND ----------

local_model = SentenceTransformer(snapshot_location, trust_remote_code=True)

example_inputs_embedded = local_model.encode(example_inputs, normalize_embeddings=True)
print(example_inputs_embedded)
print(type(example_inputs_embedded))

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyFunc wrapper model

# COMMAND ----------

class SentenceTransformerEmbeddingModel(mlflow.pyfunc.PythonModel):
  @staticmethod
  def _convert_input_to_list(model_input):
    import numpy as np
    import pandas as pd

    # If the input is a DataFrame or numpy array,
    # convert the first column to a list of strings.
    if isinstance(model_input, pd.DataFrame):
      list_input = model_input.iloc[:, 0].tolist()
    elif isinstance(model_input, np.ndarray):
      list_input = model_input[:, 0].tolist()
    else:
      assert isinstance(model_input, list),\
        f"Model expected model_input to be a pandas.DataFrame, numpy.ndarray, or list, but was given: {type(model_input)}"
      list_input = model_input
    return list_input

  def load_context(self, context):
    """
    This method initializes the model from the cached artifacts.
    """
    from sentence_transformers import SentenceTransformer

    self.model = SentenceTransformer(context.artifacts["repository"], trust_remote_code=True)

    self.model.to("cuda")

  def predict(self, context, model_input):
    """
    This method generates prediction for the given input.
    """

    # convert to a list ["sentence", "sentence", ...]
    input_texts = SentenceTransformerEmbeddingModel._convert_input_to_list(model_input)

    embeddings = self.model.encode(input_texts, normalize_embeddings=True)

    #type(embeddings) == np.ndarray
    return embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC ## Infer the signature

# COMMAND ----------

signature = mlflow.models.signature.infer_signature(example_inputs, example_inputs_embedded)
print(signature)

# COMMAND ----------

# MAGIC %md ## Log & register

# COMMAND ----------

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        mlflow_artifact_path,
        python_model=SentenceTransformerEmbeddingModel(),
        artifacts={"repository": snapshot_location},
        signature=signature,
        input_example=example_inputs,
        pip_requirements=["sentence_transformers", "transformers", "torch", "numpy", "pandas"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC  By default, MLflow registers models in the Databricks workspace model registry. To register models in Unity Catalog instead, we follow the [documentation](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html) and set the registry server as Databricks Unity Catalog.
# MAGIC
# MAGIC  In order to register a model in Unity Catalog, there are [several requirements](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements), such as Unity Catalog must be enabled in your workspace.
# MAGIC

# COMMAND ----------

registered_model = mlflow.register_model(
    model_info.model_uri,
    registered_model_name,
)

# Choose the right model version registered in the above cell.
client.set_registered_model_alias(name=registered_model_name, alias="Prod", version=registered_model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC Once the model is registered, we can use API to create a Databricks GPU Model Serving Endpoint that serves the `mixtral-8x7b-instruct` model.
# MAGIC
# MAGIC Note that the below deployment requires GPU model serving. For more information on GPU model serving, see the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html#gpu). The feature is in Public Preview.

# COMMAND ----------

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": registered_model.name,
            "model_version": registered_model.version,
            "workload_type": serving_workload_type,
            "workload_size": serving_workload_size,
            "scale_to_zero_enabled": serving_scale_to_zero_enabled
        }
    ]
})
w.serving_endpoints.create(name=endpoint_name, config=config)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the model serving endpoint is ready, you can query it.

# COMMAND ----------

browser_url = du.get_browser_hostname()

print(f"View endpoint status: https://{browser_url}/ml/endpoints/{endpoint_name}")

# Continuously check the status of the serving endpoint
while w.serving_endpoints.get(name=endpoint_name).state.ready != EndpointStateReady.READY:
    print("Endpoint is updating - can take 15 - 45 minutes. Waiting 5 mins to check again...")
    time.sleep(60*5)  # Wait for 10 seconds before checking again
