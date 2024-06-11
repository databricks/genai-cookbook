# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # Load your evaluation set from the previous step

# COMMAND ----------

df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# If you did not collect feedback from your stakeholders, and want to evaluate using a manually curated set of questions, you can use the structure below.

eval_data = [
    {
        ### REQUIRED
        # Question that is asked by the user
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "your-request-id",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": "There's no significant difference.",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "doc_uri_2_1",
            },
        ],
    }
]

# Uncomment this row to use the above data instead of your evaluation set
# eval_df = pd.DataFrame(eval_data)

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
# MAGIC ## Load the correct Python environment for the POC's app
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation on the POC app

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the evaluation results
# MAGIC
# MAGIC You can explore the evaluation results using the above links to the MLflow UI.  If you prefer to use the data directly, see the cells below.

# COMMAND ----------

# Summary metrics across the entire evaluation set
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)
