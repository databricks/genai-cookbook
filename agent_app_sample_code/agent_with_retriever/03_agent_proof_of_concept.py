# Databricks notebook source
# MAGIC %md
# MAGIC # Agent proof of concept.
# MAGIC
# MAGIC By the end of this notebook, you will have created a POC of your Agent that you can interact with, and ask questions.
# MAGIC
# MAGIC This means:
# MAGIC - We will have a mlflow model registered in the "Models" tab on the Databricks menu on the left. Models that are registered are just assets that can be instantiated from another notebook, but are not served on an endpoint. These models can be invoked with `mlflow.invoke()`.
# MAGIC - We will have a served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
%pip install -qqqq -U databricks-agents openai databricks-vectorsearch databricks-sdk mlflow mlflow-skinny
# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the global configuration

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration

# COMMAND ----------

# MAGIC %run ./agents/agent_config

# COMMAND ----------

# If you used the 02_data_pipeline notebook, you can use this cell to get the configuration of the Retriever

# TODO: Add this support

# COMMAND ----------

# TODO: This must be manually set by the user.  See TODO above for providing a way to automatically set this from 02_data_pipeline
retriever_config = RetrieverToolConfig(
    vector_search_index=f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}_chunked_docs_index",
    vector_search_schema=RetrieverSchemaConfig(
        primary_key="chunk_id",
        chunk_text="content_chunked",
        document_uri="doc_uri",
        additional_metadata_columns=[],
    ),
    parameters=RetrieverParametersConfig(num_results=5, query_type="ann"),
    vector_search_threshold=0.1,
    chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",
    prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",
    tool_description_prompt="Search for documents that are relevant to a user's query about the [REPLACE WITH DESCRIPTION OF YOUR DOCS].",
)

# TODO: Improve these docs
# `llm_endpoint_name`: Model Serving endpoint with the LLM for your Agent. 
#     - Either an [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) Provisioned Throughput / Pay-Per-Token or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/chat` with support for [function calling](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html).  Supported models: `databricks-meta-llama-3-70b-instruct` or any of the Azure OpenAI / OpenAI models.

llm_config = LLMConfig(
    # https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    # llm_endpoint_name="databricks-meta-llama-3-70b-instruct",
    llm_endpoint_name="ep-gpt4o",
    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template=(
        """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the information from tools that are explictly specified to you.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
    ),
    # Parameters that control how the LLM responds.
    llm_parameters=LLMParametersConfig(temperature=0.01, max_tokens=1500),
)

agent_config = AgentConfig(
    # TODO: Make this generalized to include multiple tools
    retriever_tool=retriever_config,
    llm_config=llm_config,
    input_example={
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
)

agent_config.dict()

# COMMAND ----------

# MAGIC %run ./validators/validate_agent_config

# COMMAND ----------

validate_retriever_config(retriever_config)
validate_llm_config(llm_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiement name
# MAGIC
# MAGIC Used to store the Agent's model

# COMMAND ----------

import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# Use OpenAI client with Model Serving
# TODO: Improve the docs for why this happens
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN
os.environ["DATABRICKS_HOST"] = f"{API_ROOT}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the Agent's code

# COMMAND ----------

# MAGIC %run ./agents/function_calling_agent_w_retriever_tool

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iteratively vibe check & adjust the Agent's configuration
# MAGIC
# MAGIC If you need to adjust the Agent's code, you can do so in the `./agents/function_calling_agent_w_retriever_tool` Notebook

# COMMAND ----------

# Load the Agent
agent = AgentWithRetriever(agent_config=agent_config.dict())
agent.load_context(None)

# Example for testing multiple turns of converastion

# TODO: Show how to run Agent Evaluation here to help with Vibe checking

# 1st turn of converastion
first_turn_input = {
    "messages": [
        {"role": "user", "content": f"what is lakehouse monitoring?"},
    ]
}

response = agent.predict(model_input=first_turn_input)
print(response["content"])

print()
print("------")
print()

# 2nd turn of converastion
new_messages = response["messages"]
new_messages.append({"role": "user", "content": f"how do i use it?"})
# print(type(new_messages))
second_turn_input = {"messages": new_messages}
response = agent.predict(model_input=second_turn_input)
print(response["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Agent POC to the Review App

# COMMAND ----------

# TODO: Remove the need for this w/ automatic credential support

w = WorkspaceClient()

# Where to save the secret
SCOPE_NAME = "lilac"
SECRET_NAME = "nst_pat"

# PAT token
SECRET_TO_SAVE = "" # REMOVED

existing_scopes = [scope.name for scope in w.secrets.list_scopes()]
if SCOPE_NAME not in existing_scopes:
    print(f"Creating secret scope `{SCOPE_NAME}`")
    w.secrets.create_scope(scope=SCOPE_NAME)
else:
    print(f"Secret scope `{SCOPE_NAME}` exists")

existing_secrets = [secret.key for secret in w.secrets.list_secrets(scope=SCOPE_NAME)]
if SECRET_NAME not in existing_secrets:
    print(f"Saving secret to `{SCOPE_NAME}.{SECRET_NAME}`")
    w.secrets.put_secret(scope=SCOPE_NAME, key=SECRET_NAME, string_value=SECRET_TO_SAVE)
else:
    print(f"Secret named `{SCOPE_NAME}.{SECRET_NAME}` already exists")


# COMMAND ----------

import pkg_resources

def get_package_version(package_name):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        return package_version
    except pkg_resources.DistributionNotFound:
        return f"{package_name} is not installed"

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
import yaml
import openai
from databricks import agents
from databricks import vector_search

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=llm_config.llm_endpoint_name),
    # TODO: Add the embedding model here
    DatabricksVectorSearchIndex(index_name=retriever_config.vector_search_index)
]

# Specify the full path to the Agent notebook 
model_file = "agents/function_calling_agent_w_retriever_tool"
model_path = os.path.join(os.getcwd(), model_file)

# Dump the config so the agent can use it for testing locally
# TODO: This should be automatically handled by MLflow - for some reason, MLflow doesn't inject the logged configuraiton into the model when loading the model locally with mlflow.pyfunc.load_model(model_info.model_uri)
chain_config_filepath = 'agents/generated_configs/agent.yaml'
with open(chain_config_filepath, 'w') as f:
  yaml.dump(agent_config.dict(), f)

__mlflow_model_config__ = agent_config.dict()

with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        # model_config=agent_config.dict(), # DOES NOT WORK
        model_config = os.path.join(os.getcwd(), chain_config_filepath),  # DOES NOT WORK EITHER
        artifact_path="agent",
        input_example=agent_config.input_example,
        resources=databricks_resources,
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
        # specify all python packages that are required by your Agent
        pip_requirements=[
            "openai==" + openai.__version__,
            "databricks-agents==" + agents.__version__,
            "databricks-vectorsearch==" + vector_search.__version__,
        ],
    )

# COMMAND ----------

### Invoke the logged model
model = mlflow.pyfunc.load_model(model_info.model_uri)
model.predict(agent_config.input_example)

# COMMAND ----------

# Use Unity Catalog as the model registry
mlflow.set_registry_uri('databricks-uc')

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=model_info.model_uri, 
                                                 name=UC_MODEL_NAME)

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(UC_MODEL_NAME, 
                                uc_registered_model_info.version,
                                environment_vars={"DATABRICKS_HOST" : 'https://' + mlflow.utils.databricks_utils.get_workspace_url(), 
                                                  "DATABRICKS_TOKEN": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}"}
                                )

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")
