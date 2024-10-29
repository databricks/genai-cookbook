# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC
# MAGIC Use this notebook to deploy a version of the Agent you registered to Unity Catalog with `02_agent`.
# MAGIC
# MAGIC By the end of this notebook, you will have deployed a version of your Agent that you can interact with and share with your business stakeholders for feedback, even if they don't have access to your Databricks workspace.
# MAGIC
# MAGIC This means we will have:
# MAGIC - A served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC - Agent Evaluation's Review Application connected to your served model.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Important note:** Throughout this notebook, we indicate which cell's code you:
# MAGIC - ✅✏️ should customize - these cells contain code & config with business logic that you should edit to meet your requirements & tune quality.
# MAGIC - 🚫✏️ should not customize - these cells contain boilerplate code required to load/save/execute your Agent
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!  You CAN change these cells, but if this is the first time using this notebook, we suggest not doing so.*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ Install Python libraries
# MAGIC
# MAGIC You do not need to modify this cell unless you need additional Python packages in your Agent.

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
%pip install -qqqq -U \
  -r requirements.txt `# shared packages` \
  langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 `# Required if using the LangChain SDK notebooks`
# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC To use this notebook, all you need to do:
# MAGIC 1. Set the below `uc_model_version_number` variable to identify the version of the Unity Catalog model you want to deploy.
# MAGIC   - This version number is printed out by the `02_agent` notebook OR you can navigate to the Agent's model in Unity Catalog to select a version.
# MAGIC 2. Press `Run All`

# COMMAND ----------

uc_model_version_number = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0️⃣ Load the Agent's configuration that is shared with the other notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ (Optional) Change Agent's shared storage location configuration
# MAGIC
# MAGIC ** If you configured `00_shared_config`, just run this cell as-is.**
# MAGIC
# MAGIC From the shared configuration, this notebook uses:
# MAGIC * The Unity Catalog model name
# MAGIC
# MAGIC *These values can be set here if you want to use this notebook independently.*

# COMMAND ----------

from utils.cookbook.agent_config import CookbookAgentConfig

# Load the shared configuration
cookbook_shared_config = CookbookAgentConfig.from_yaml_file('./configs/cookbook_config.yaml')

# Print configuration 
cookbook_shared_config.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Deploy the Agent

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 🚫✏️ Deploy the Model Serving endpoint & Review App

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Get the Workspace client
w = WorkspaceClient()

# Deploy to enable the Review App and create an API endpoint
deployment_info = agents.deploy(cookbook_shared_config.uc_model, uc_model_version_number)

# Wait for the deployment to be ready 
# This code continually polls the endpont for its status and will complete once the endpoint & Review App are ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅✏️ Provide instructions to your stakeholders
# MAGIC
# MAGIC The below cell loads instructions to your stakeholders that they will see within the Review App.

# COMMAND ----------

from databricks import agents

app_name = "Your App Name"
instructions_to_reviewer = f"""## Instructions for Testing **{app_name}**

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing {app_name}. Your contributions are essential to delivering a high-quality product to our end users."""

print(instructions_to_reviewer)

agents.set_review_instructions(cookbook_shared_config.uc_model, instructions_to_reviewer)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ View the inference tables
# MAGIC
# MAGIC The deployed Agent's Inference Tables contain logs for every request sent to the REST API or Review App.
# MAGIC
# MAGIC There are 3 tables:
# MAGIC 1. **Raw logs**: contains the raw logs collected by Model Serving.  Most users will not need to use this table, but it is available in the case that the Databricks-run job to create request/assessment logs fails for a row.
# MAGIC 2. **Request logs**: contains one row per user query, post-processed from the raw logs.  *Note that if a converastion contains multiple turns of chat, there will be one row per chat turn.*. In this table, you will find:
# MAGIC   - request
# MAGIC   - response
# MAGIC   - MLflow trace
# MAGIC   - other metadata e.g., timestamp, etc
# MAGIC 3. **Assessment logs**: contains user feedback from the Review App, post-processed from the raw logs.  *Note this table contains one row for every feedback action e.g., if a user presses thumbs up and then selects a reason why, there will be 2 rows.  The `04_create_evaluation_set` notebook contains code to merge these multiple rows into a single asessment per user/request.*
# MAGIC   - user email
# MAGIC   - thumbs up / down
# MAGIC   - reasons for thumbs up / down
# MAGIC   - free text comments
# MAGIC   - thumbs up / down for each retrieved document
# MAGIC
# MAGIC *Note: To view the MLflow Traces in the request log, run `display(spark.table("catalog.schema.table_name"))` in a notebook and click on a row.  The trace will be displayed in the notebook.*
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC 📢❗❗ **Important**
# MAGIC
# MAGIC It can take up 2 hours for requests to be loaded into the raw logs.  A job to post-process the raw logs into the request and assessment logs runs every hour.  
# MAGIC
# MAGIC The request and assessment logs will NOT be created until 1+ request is in the raw logs and the post-processing job runs.

# COMMAND ----------

from utils.get_inference_tables import get_inference_tables
from utils.cookbook.url_utils import get_table_url
from IPython.display import display_markdown

inference_tables = get_inference_tables(cookbook_shared_config.uc_model)

inference_logs_info = f"""
- Raw logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['raw_payload_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['raw_payload_logs']}")})
- Request logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['request_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['request_logs']}")})
- Feedback logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['assessment_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['assessment_logs']}")})
"""

display_markdown(inference_logs_info, raw=True)
