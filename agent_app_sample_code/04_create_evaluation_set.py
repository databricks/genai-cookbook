# Databricks notebook source
# MAGIC %md # Review App Logs to Evaluation Set
# MAGIC
# MAGIC This notebook will bootstrap an evaluation set with the feedback that stakeholders have provided by using the Review App.
# MAGIC
# MAGIC Note that you can bootstrap an evaluation set with *just* questions, so even if your stakeholders have only chatted with the app vs. providing feedback, you can follow this step.
# MAGIC
# MAGIC Visit [documentation](https://docs.databricks.com/generative-ai/agent-evaluation/evaluation-set.html#evaluation-set-schema) to understand the Agent Evaluation Evaluation Set schema - these fields are referenced below.
# MAGIC
# MAGIC At the end of this step, you will have an Evaluation Set that contains:
# MAGIC
# MAGIC 1. Requests with a 👍 :
# MAGIC    - `request`: As entered by the user
# MAGIC    - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
# MAGIC 2. Requests with a 👎 :
# MAGIC    - `request`: As entered by the user
# MAGIC    - `expected_response`: If the user edited the response, that is used, otherwise, null.
# MAGIC 3. Requests without any feedback e.g., no 👍 or 👎
# MAGIC    - `request`: As entered by the user
# MAGIC
# MAGIC Across all of the above, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.
# MAGIC
# MAGIC **Once you have run this notebook, return to the `02_agent` notebook to evaluate the quality of your agent using your new evaluation set!**

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
# MAGIC ### 🚫✏️ Install Python libraries
# MAGIC
# MAGIC You do not need to modify this cell unless you need additional Python packages in your Agent.

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0️⃣ Setup: Load the Agent's configuration that is shared with the other notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ Get the shared configuration
# MAGIC
# MAGIC ** If you configured `00_shared_config`, just run this cell as-is.**
# MAGIC
# MAGIC From the shared configuration, this notebook uses:
# MAGIC * The Evaluation Set stored in Unity Catalog
# MAGIC * The MLflow experiment for tracking Agent verions & their quality evaluations
# MAGIC * The UC model to get details of the deployed Agent
# MAGIC
# MAGIC *These values can be set here if you want to use this notebook independently.*

# COMMAND ----------

from cookbook_utils.cookbook_config import AgentCookbookConfig
import mlflow

# Load the shared configuration
cookbook_shared_config = AgentCookbookConfig.from_yaml_file('./configs/cookbook_config.yaml')

# Print configuration 
cookbook_shared_config.pretty_print()

# Set the MLflow Experiment that is used to track metadata about each run of this Data Pipeline.
experiment_info = mlflow.set_experiment(cookbook_shared_config.mlflow_experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 🚫✏️ Import the cookbook utilities for transforming inference tables into an evaluation set

# COMMAND ----------

# MAGIC %run ./utils/eval_set_utilities

# COMMAND ----------

# MAGIC %md ## 🚫✏️ Get the request and assessment log tables
# MAGIC
# MAGIC These tables are updated every ~hour with data from the raw Inference Table. See [docs](https://docs.databricks.com/en/generative-ai/deploy-agent.html#agent-enhanced-inference-tables) for the schema.

# COMMAND ----------

from cookbook_utils.get_inference_tables import get_inference_tables

inference_table_locations = get_inference_tables(cookbook_shared_config.uc_model)
inference_table_locations = get_inference_tables("main.eric_peter_agents.agent_app_name_model1")

# Get the table names
assessment_log_table_name = f"{inference_table_locations['uc_catalog_name']}.{inference_table_locations['uc_schema_name']}.`{inference_table_locations['table_names']['assessment_logs']}`"
request_log_table_name = f"{inference_table_locations['uc_catalog_name']}.{inference_table_locations['uc_schema_name']}.`{inference_table_locations['table_names']['request_logs']}`"

print(f"Assessment logs: {assessment_log_table_name}")
print(f"Request logs: {request_log_table_name}")

# De-duplicate the assessment logs
assessment_log_df = _dedup_assessment_log(spark.table(assessment_log_table_name))
request_log_df = spark.table(request_log_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚫✏️ ETL the request & assessment logs into Evaluation Set schema
# MAGIC
# MAGIC Note: We leave the complete set of columns from the request and assesment logs in this table - you can use these for debugging any issues.

# COMMAND ----------

requests_with_feedback_df = create_potential_evaluation_set(
    request_log_df, assessment_log_df
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅✏️  Inspect the potential evaluation set using MLflow Tracing
# MAGIC
# MAGIC Click on the `trace` column in the displayed table to view the Trace.  You should inspect these records and determine which should be included in your evaluation set.

# COMMAND ----------

from pyspark.sql import functions as F
display(
    requests_with_feedback_df.select(
        F.col("request_id"),
        F.col("request"),
        F.col("response"),
        F.col("trace"),
        F.col("expected_response"),
        F.col("expected_retrieved_context"),
        F.col("is_correct"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅✏️ Save the resulting evaluation set to a Delta Table
# MAGIC
# MAGIC Based on your analysis above, save the selected subset of records to your evaluation set.

# COMMAND ----------

eval_set = requests_with_feedback_df[
    [
        "request",
        "request_id",
        "expected_response",
        "expected_retrieved_context",
        "source_user",
        "source_tag",
    ]
]

eval_set.write.format("delta").mode("overwrite").saveAsTable(cookbook_shared_config.evaluation_set_table)

display(spark.table(cookbook_shared_config.evaluation_set_table))
