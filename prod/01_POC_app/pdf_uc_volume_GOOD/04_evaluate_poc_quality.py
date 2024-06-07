# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-rag-studio mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import rag_studio
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md # Turn logs from the Review App into an Evaluation Set
# MAGIC
# MAGIC Here, we ETL the logs from the Review App into an Evaluation Set.  It is important to review each row and ensure the data quality is high e.g., the question is logical and the response makes sense.
# MAGIC
# MAGIC 1. Requests with a 👍 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
# MAGIC 2. Requests with a 👎 :
# MAGIC     - `request`: As entered by the user
# MAGIC     - `expected_response`: If the user edited the response, that is used, otherwise, null.
# MAGIC 3. Requests without any feedback
# MAGIC     - `request`: As entered by the user
# MAGIC
# MAGIC Across all types of requests, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Delta Tables with the Review App's logs
# MAGIC
# MAGIC Every ~hour, the raw payload log is unpacked into a cleanly formatted `assessment` and `request_log` Delta Tables.  This gets the names of those tables.

# COMMAND ----------

# # Get the name of the Inference Tables where logs are stored
# active_deployments = rag_studio.list_deployments()
# active_deployment = next((item for item in active_deployments if item.model_name == UC_MODEL_NAME), None)

# endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

# try:
#     endpoint_config = endpoint.config.auto_capture_config
# except AttributeError as e:
#     endpoint_config = endpoint.pending_config.auto_capture_config

# inference_table_name = endpoint_config.state.payload_table.name
# inference_table_catalog = endpoint_config.catalog_name
# inference_table_schema = endpoint_config.schema_name

# # Cleanly formatted tables
# assessment_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
# request_table = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"

# print(f"Assessment logs: {assessment_table}")
# print(f"Request logs: {request_table}")

#tempppppp
assessment_table = f'rag.rag_e.`rag_studio-finance_bench_new_payload_assessment_logs`'
request_table = f'rag.rag_e.`rag_studio-finance_bench_new_payload_request_logs`'

requests_df = spark.table(request_table)
assessment_df = deduplicate_assessments_table(assessment_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the request logs using MLflow Tracing
# MAGIC
# MAGIC Click on each trace in the displayed table to view it.

# COMMAND ----------


requests_with_feedback_df = requests_df.join(assessment_df, requests_df.databricks_request_id == assessment_df.request_id, "left")
display(requests_with_feedback_df.select("request_raw", "trace", "source", "text_assessment", "retrieval_assessments"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the assessment logs into the evaluation set

# COMMAND ----------

# TODO: Make this work!!!!!

requests_with_feedback_df.createOrReplaceTempView('latest_assessments')
evaluation_set = spark.sql(f"""
-- Thumbs up.  Use the model's generated response as the expected_response
select
  a.request_id,
  r.request,
  r.response as expected_response,
  'thumbs_up' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value == "positive"
union all
  --Thumbs down.  If edited, use that as the expected_response.
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'thumbs_down' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value = "negative"
union all
  -- No feedback.  Include the request, but no expected_response
select
  a.request_id,
  r.request,
  IF(
    a.text_assessment.suggested_output != "",
    a.text_assessment.suggested_output,
    NULL
  ) as expected_response,
  'no_feedback_provided' as type,
  a.source.id as user_id
from
  latest_assessments as a
  join {request_table} as r on a.request_id = r.databricks_request_id
where
  a.text_assessment.ratings ["answer_correct"].value != "negative"
  and a.text_assessment.ratings ["answer_correct"].value != "positive"
  """)
display(evaluation_set)

# Save to Pandas DF
eval_df = evaluation_set.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC TEMP CELL
# MAGIC
# MAGIC # TODO: Add cell to write the evaluation set to Delta Table in EVALUATION_SET_FQN

# COMMAND ----------

df = spark.table("rag.financebench.finance_bench_eval_set")
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# Load manually
eval_data = [
    {
        # Optional, user specified to identify each row
        "request_id": "your-request-id",
        # Question that is asked by the user
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        # Optional: correct response to the question
        # If provided, Quality Lab can compute additional metrics.
        "expected_response": "There's no significant difference.",
        # Optional: Which documents should be retrieved.
        # If provided, Quality Lab can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "doc_uri_2_1",
            },
        ],
    }
]

eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the POC application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_CHAIN_RUN_NAME}'", output_format="list")

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the correct Python environment for the model
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-rag",
    )
