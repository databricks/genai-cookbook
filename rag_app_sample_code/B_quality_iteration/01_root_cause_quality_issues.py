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

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

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
# MAGIC ## Root cause analysis (ground truth available)
# MAGIC
# MAGIC Based on your evaluation set, either run this cell or the following cell.  Do NOT run both.

# COMMAND ----------


# Define the conditions and corresponding root cause and overall rating
conditions = [
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "yes"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "yes") ,
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "yes"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),

    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "no") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),
    
    (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
    (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/correctness/rating"] == "yes") & 
    (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes")
]

root_causes = [
    "Improve Retrieval",
    "Improve Retrieval",
    "Improve Retrieval",
    "Improve Retrieval",
    "Improve Retrieval",
    "No root cause, record passed",
    "Improve Generation",
    "Improve Generation",
    "Improve Generation",
    "Improve Generation",
    "No root cause, record passed",
]

overall_ratings = [
    "fail",
    "fail",
    "fail",
    "fail",
    "fail",
    "pass",
    "fail",
    "fail",
    "fail",
    "fail",
    "pass"
]

root_cause_rationales = [
    "Retrieval is poor.",
    "LLM generates relevant response, but retrieval is poor e.g., the LLM ignores retrieval and uses its training knowledge to answer.",
    "Retrieval quality is poor, but LLM gets the answer correct regardless.",
    "Response is grounded in retrieval, but retrieval is poor.",
    "Relevant response grounded in the retrieved context, but retrieval may not be related to the expected answer.",
    "Retrieval finds enough information for the LLM to correctly answer.",
    "Hallucination",
    "Hallucination, correct but generates details not in context",
    "Good retrieval, but the LLM does not provide a relevant response.",
    "Good retrieval and relevant response, but not correct.",
    "No issues."
]


# Create new columns in the dataframe based on the conditions
eval_results_df["overall/root_cause/rating"] = pd.Series(pd.NA, index=eval_results_df.index)
eval_results_df["overall/assessment"] = pd.Series(pd.NA, index=eval_results_df.index)
eval_results_df["overall/root_cause/rationale"] = pd.Series(pd.NA, index=eval_results_df.index)

for condition, root_cause, overall_rating,root_cause_rationale  in zip(conditions, root_causes, overall_ratings, root_cause_rationales):
    eval_results_df.loc[condition, "overall/root_cause/rating"] = root_cause
    eval_results_df.loc[condition, "overall/assessment"] = overall_rating
    eval_results_df.loc[condition, "overall/root_cause/rationale"] = root_cause_rationale

# COMMAND ----------

# MAGIC %md
# MAGIC ## Root cause analysis (ground truth NOT available)
# MAGIC
# MAGIC Based on your evaluation set, either run this cell or the previous cell.  Do NOT run both.

# COMMAND ----------

# # Define the conditions and corresponding root cause and overall rating
# conditions = [
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),
    
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),
    
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),
    
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]<.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),

    
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),
    
#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "no") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),

#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "no"),

#     (eval_results_df["retrieval/llm_judged/chunk_relevance/precision"]>=.5) & 
#     (eval_results_df["response/llm_judged/groundedness/rating"] == "yes") & 
#     (eval_results_df["response/llm_judged/relevance_to_query/rating"] == "yes"),
# ]

# root_causes = [
#     "Improve Retrieval",
#     "Improve Retrieval",
#     "Improve Retrieval",
#     "Improve Retrieval",
#     "Improve Generation",
#     "Improve Generation",
#     "Improve Generation",
#     "No root cause, record passed",
# ]

# overall_ratings = [
#     "fail",
#     "fail",
#     "fail",
#     "pass",
#     "fail",
#     "fail",
#     "fail",
#     "pass"
# ]

# root_cause_rationales = [
#     "Retrieval quality is poor.",
#     "Retrieval quality is poor.",
#     "Response is grounded in retrieval, but retrieval is poor.",
#     "Relevant response grounded in the retrieved context and relevant, but retrieval is poor.",
#     "Hallucination",
#     "Hallucination",
#     "Good retrieval & grounded, but LLM does not provide a relevant response.",
#     "Good retrieval and relevant response. Collect ground-truth to know if the answer is correct.",
# ]


# # Create new columns in the dataframe based on the conditions
# eval_results_df["overall/root_cause/rating"] = pd.Series(pd.NA, index=eval_results_df.index)
# eval_results_df["overall/assessment"] = pd.Series(pd.NA, index=eval_results_df.index)
# eval_results_df["overall/root_cause/rationale"] = pd.Series(pd.NA, index=eval_results_df.index)

# for condition, root_cause, overall_rating,root_cause_rationale  in zip(conditions, root_causes, overall_ratings, root_cause_rationales):
#     eval_results_df.loc[condition, "overall/root_cause/rating"] = root_cause
#     eval_results_df.loc[condition, "overall/assessment"] = overall_rating
#     eval_results_df.loc[condition, "overall/root_cause/rationale"] = root_cause_rationale

# COMMAND ----------

# MAGIC %md
# MAGIC # Determine the frequency of each root cause.

# COMMAND ----------

root_cause_frequencies = eval_results_df["overall/root_cause/rating"].value_counts(normalize=True) * 100
root_cause_rationale_frequencies = eval_results_df["overall/root_cause/rationale"].value_counts(normalize=True) * 100

print("Root cause frequencies: \n")
print(root_cause_frequencies)
print("\n-------\n")
print("Root cause rationale frequencies: \n")
print(root_cause_rationale_frequencies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the traces to develop intuition about the issues

# COMMAND ----------

columns_to_display = [
    "request_id",
    "overall/root_cause/rating",
    "overall/assessment",
    "overall/root_cause/rationale",
    "trace",
    "expected_retrieved_context",
    "expected_response",
    "response/llm_judged/relevance_to_query/rating",
    "response/llm_judged/relevance_to_query/rationale",
    "response/llm_judged/groundedness/rating",
    "response/llm_judged/groundedness/rationale",
    "response/llm_judged/correctness/rating",
    "response/llm_judged/correctness/rationale",
    "retrieval/llm_judged/chunk_relevance/ratings",
    "retrieval/llm_judged/chunk_relevance/rationales",
    "retrieval/llm_judged/chunk_relevance/precision",
    "chain/total_input_token_count",
    "chain/total_output_token_count",
    "chain/total_token_count",
    "chain/latency_seconds",
]

# COMMAND ----------

# MAGIC %md
# MAGIC # Debugging the issue
# MAGIC
# MAGIC TODO: Add link to step 5.1.1 and 5.1.2

# COMMAND ----------

# MAGIC %md
# MAGIC ### Debugging retrieval issues
# MAGIC

# COMMAND ----------

display(eval_results_df[eval_results_df["overall/root_cause/rating"] == "Improve Retrieval"][columns_to_display])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Debugging generation issues

# COMMAND ----------

display(eval_results_df[eval_results_df["overall/root_cause/rating"] == "Improve Generation"][columns_to_display])
