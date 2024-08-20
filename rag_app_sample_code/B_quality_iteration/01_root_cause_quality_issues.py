# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
import pandas as pd

# COMMAND ----------

# MAGIC %run ../00_global_config

# COMMAND ----------

# MAGIC %md # Load the evaluation data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_CHAIN_RUN_NAME}'", output_format="list")

# if len(runs) != 1:
#     raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the evaluation data

# COMMAND ----------

mc = mlflow.MlflowClient()
eval_results_df = mc.load_table(experiment_id=poc_run.info.experiment_id, run_ids=[poc_run.info.run_id], artifact_file="eval_results.json")
display(eval_results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Root cause analysis
# MAGIC
# MAGIC Below you will see a few examples of how to link the evaluation results back to potential root causes and their corresponding fixes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Find requests that are incorrect and have low retrieval accuracy. Potential fix: improve the retriever

# COMMAND ----------

display(eval_results_df[(eval_results_df["response/llm_judged/correctness/rating"]=="no") & (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5)])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Find requests that are not grounded even though retrieval accuracy is high. Potential fix: tune the generator prompt to avoid hallucinations.

# COMMAND ----------

display(eval_results_df[(eval_results_df["response/llm_judged/groundedness/rating"]=="no") & (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5)])
