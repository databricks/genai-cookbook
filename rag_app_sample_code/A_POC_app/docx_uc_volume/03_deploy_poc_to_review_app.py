# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import time
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md ## Log the chain to MLflow & test the RAG chain locally
# MAGIC
# MAGIC This will save the chain using MLflow's code-based logging and invoke it locally to test it.  
# MAGIC
# MAGIC **MLflow Tracing** allows you to inspect what happens inside the chain.  This same tracing data will be logged from your deployed chain along with feedback that your stakeholders provide to a Delta Table.
# MAGIC
# MAGIC `# TODO: link docs for code-based logging`

# COMMAND ----------

# Log the model to MLflow
# TODO: remove example_no_conversion once this papercut is fixed
with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), CHAIN_CODE_FILE
        ),  # Chain code file e.g., /path/to/the/chain.py
        model_config=rag_chain_config,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=rag_chain_config[
            "input_example"
        ],  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
    )

    # Attach the data pipeline's configuration as parameters
    mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

    # Attach the data pipeline configuration 
    mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the chain locally

# COMMAND ----------

chain_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is RAG?", # Replace with a question relevant to your use case
        }
    ]
}
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(chain_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to the Review App
# MAGIC
# MAGIC Now, let's deploy the POC to the Review App so your stakeholders can provide you feedback.
# MAGIC
# MAGIC Notice how simple it is to call `agents.deploy()` to enable the Review App and create an API endpoint for the RAG chain!

# COMMAND ----------

instructions_to_reviewer = f"""## Instructions for Testing the {RAG_APP_NAME}'s Initial Proof of Concept (PoC)

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing {RAG_APP_NAME}. Your contributions are essential to delivering a high-quality product to our end users."""

print(instructions_to_reviewer)

# COMMAND ----------

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

# Add the user-facing instructions to the Review App
agents.set_review_instructions(UC_MODEL_NAME, instructions_to_reviewer)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant stakeholders access to the Review App
# MAGIC
# MAGIC Now, grant your stakeholders permissions to use the Review App.  Your stakeholders do not Databricks accounts as long as you have [insert docs].
# MAGIC
# MAGIC `#TODO: add docs link`

# COMMAND ----------

user_list = ["eric.peter@databricks.com"]

# Set the permissions.  If successful, there will be no return value.
agents.set_permissions(model_name=UC_MODEL_NAME, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

# COMMAND ----------

# MAGIC %md ## Optional: Find review app name
# MAGIC
# MAGIC If you lose this notebook's state and need to find the URL to your Review App, run this cell.
# MAGIC
# MAGIC Alternatively, you can construct the Review App URL as follows:
# MAGIC
# MAGIC `https://<your-workspace-url>/ml/reviews/{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}/{UC_MODEL_VERSION_NUMBER}/instructions`

# COMMAND ----------

active_deployments = agents.list_deployments()

active_deployment = next((item for item in active_deployments if item.model_name == UC_MODEL_NAME), None)

print(f"Review App URL: {active_deployment.review_app_url}")
