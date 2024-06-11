# Databricks notebook source
# MAGIC %md # Evaluate potential fixes
# MAGIC
# MAGIC This notebook evaluates different data pipelines, chain configurations, and chain code fixes.
# MAGIC
# MAGIC **Inputs**
# MAGIC * 1+ fix to evaluate, can be any combination of:
# MAGIC   * Data pipeline fix(es)
# MAGIC   * Chain configuration fix(es)
# MAGIC   * Chain code fix(es)
# MAGIC * Baseline chain to compare against, stored as an MLflow run alongside its evaluation
# MAGIC     * Initially this will be your POC chain, but once you identify a higher-quality experiment, this will become your new baseline
# MAGIC * List of Agent Evaluation metrics to use in selecting the best app
# MAGIC
# MAGIC **What happens:**
# MAGIC Each experiment is used to modify the baseline chain.  For example, if your baseline chain had {param1: xx, param2: yy} and the experiment had {param2: zz}, the resulting experiment would be {param1: xx, param2: zz} .  If you provided 2 chain configs, 3 data pipelines, and 4 chain code files, 2 + 3 + 4 = 9 fixes will be evaluated.
# MAGIC
# MAGIC **Outputs**
# MAGIC 1. For every fix
# MAGIC     * Quality/cost/latency metrics (from Mosiac AI Agent Evaluation)
# MAGIC     * MLflow logged model, ready to deploy to production or the Review App
# MAGIC 2. "Winning" fix (or that the baseline is the best) 
# MAGIC     * e.g., which fix had the highest metrics
# MAGIC
# MAGIC You can then deploy the "winning" fix to the Review App so you your stakeholders can test it -- or -- if your production quality bar is met, deploy it to a scalable, secure REST API hosted by Databricks.

# COMMAND ----------

# MAGIC %pip install -U -qqqq pyyaml databricks-agents mlflow mlflow-skinny databricks-sdk flashrank==0.2.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks import agents
import flashrank
import time
import os
import yaml
import json
import operator
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %run ../00_global_config

# COMMAND ----------

# MAGIC %run ./z_shared_utilities

# COMMAND ----------

# MAGIC %md # Baseline chain

# COMMAND ----------

# MLflow Run name containing the baseline chain + its evaluation metrics
BASELINE_CHAIN_MLFLOW_RUN_NAME = POC_CHAIN_RUN_NAME

# COMMAND ----------

# Load the baseline chain
baseline_run = get_mlflow_run(experiment_name=MLFLOW_EXPERIMENT_NAME, run_name=BASELINE_CHAIN_MLFLOW_RUN_NAME)

# Save the baseline config to YAML file
baseline_chain_config = yaml.safe_load(mlflow.artifacts.load_text(baseline_run.info.artifact_uri + "/chain/MLmodel"))['flavors']['python_function']['config']
write_baseline_chain_config_to_yaml(baseline_chain_config)

# Save the baseline chain's code to a notebook
baseline_chain_code = mlflow.artifacts.load_text(baseline_run.info.artifact_uri + "/chain/model.py")
baseline_chain_code_file_name = "baseline_chain/chain.py"
write_baseline_chain_code_to_notebook(baseline_chain_code, workspace_client=w, save_folder="baseline_chain", save_file_name="chain.py")

# Get the baseline chain's data pipeline configuration
baseline_data_pipeline_config = mlflow.artifacts.load_dict(f"{baseline_run.info.artifact_uri}/data_pipeline_config.json")

# COMMAND ----------

# MAGIC %md # Experiment setup

# COMMAND ----------

# MAGIC %md ## Data pipelines
# MAGIC
# MAGIC 1. Follow the steps in http://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-2-data-pipeline.html to implement a new data pipeline
# MAGIC 2. Put a list of the resulting MLflow *run names* in the `DATA_PIPELINE_FIXES_RUN_NAMES` variable

# COMMAND ----------

DATA_PIPELINE_FIXES_RUN_NAMES = [] # Put data pipeline MLflow run names in this array


# COMMAND ----------

# MAGIC %md ## Chain configuration
# MAGIC
# MAGIC Update the `CHAIN_CONFIG_FIXES` dictionary to specify the different configurations that you want to evaluate.  Each configuration will be referred to by it's top level key.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```
# MAGIC CHAIN_CONFIG_FIXES = {
# MAGIC     "config_1": {...},
# MAGIC     "config_2": {...}
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Each of the configurations will be used to modify the configuration present in 
# MAGIC the baseline chain's configuration.  For example, if you have:
# MAGIC
# MAGIC > `baseline_chain_config = {'a': {'x': 1, 'y': 2}}`
# MAGIC
# MAGIC > `CHAIN_CONFIG_FIXES['config_1'] = {'a': {'x': 4}}`
# MAGIC
# MAGIC The resulting configuration to evaluate will be:
# MAGIC > `experiment_to_run = {'a': {'x': 4, 'y': 2}}`
# MAGIC
# MAGIC We have included a few common strategies to evaluate below.  You can combine multiple strategies into a single strategy.  Refer to the cookbook for guidance on which strategies make sense to test based on the root cause of your quality issues.

# COMMAND ----------

CHAIN_CONFIG_FIXES = {
    # Below are a few example chain configuration fixes.  You can (and should) add your own fix based on the root cuase you have identified.
    "different_llm": {
        "databricks_resources": {
            # Replace this with another Databricks Model Serving endpoint name, such as an OpenAI External Model endpoint.
            # See the ../../Helpers/Create_OpenAI_External_Model for instructions on how to create this endpoint if you don't have one already
            # You can also try `databricks-meta-llama-3-70b-instruct` or `databricks-mixtral-8x7b-instruct` which are hosted as pay-per-token models on FMAPI
            "llm_endpoint_name": "databricks-meta-llama-3-70b-instruct",
        }
    },
    "different_llm_params": {
        "llm_config": {
            # Parameters that control how the LLM responds.  For a full list of available parameters & what they impact, see https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
            "llm_parameters": {
                "temperature": 0.01,  # The sampling temperature. 0 is deterministic and higher values introduce more randomness. Float in [0,2]
                "max_tokens": 100,  # The maximum number of tokens to generate. Integer greater than zero or None, which represents infinity
                "top_p": 1.0,  # The probability threshold used for nucleus sampling. Float in (0,1]
                "top_k": None,  # Integer greater than zero or None, which represents infinity
                "stop": [],  # Model stops generating further tokens when any one of the sequences in stop is encountered.  String or List[String]
            },
        }
    },
    "more_prompt_instructions": {
        "llm_config": {
            # As a toy example, we add "Every time you respond, say "I love Databricks".  Explain every concept like the user is 5 years old." to the default prompt.
            "llm_system_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.  Every time you respond, say "I love Databricks".  Explain every concept like the user is 5 years old.

Context: {context}""".strip(),
        },
    },
    "change_context_format": {
        "retriever_config": {
            # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.
            # {document_uri} and {chunk_text} are available.
            "chunk_template": "Important piece of information: {chunk_text} from this document: {document_uri}\n",
        }
    },
    "retriever_config": {
        "retriever_config": {
            "parameters": {
                # Number of search results that the retriever returns.  Increase or decrease.
                "k": 5,
                # Type of search to run
                # Semantic search: `ann`
                # Hybrid search (keyword + sementic search): `hybrid`
                "query_type": "ann",
            },
        },
    },
    "cite_sources": {  # Use this template to ask your bot to cite its' sources in its' response.
        "retriever_config": {
            # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.
            # {document_uri} and {chunk_text} are available.
            # If you need additional variables, you can modify the chain's code.
            "chunk_template": "Source document: {document_uri}\nPassage: {chunk_text}\n\n",
        },
        "llm_config": {
            # Ask the LLM to cite its' sources
            "llm_system_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.  Please cite your sources for each part of your answer.

Context: {context}""".strip(),
        },
    },
    "cot_reasoning": {
        "llm_config": {
            # Ask the LLM to reason before responding
            "llm_system_prompt_template": """You are an assistant that answers questions.  Before responding, you need to think step-by-step through how each piece of context provided is or isn't relevant and what specific facts from the context are useful in answering the user's question.  
            
Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.  Please cite your sources for each part of your answer.

Context: {context}""".strip(),
        },
    },
    # "safety_guardrails": {
    #     # This technique ONLY works on Databricks FMAPI pay-per-token models.
    #     # It requires enrolling the Guardrails Private Preview: https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks
    #     # `databricks-meta-llama-3-70b-instruct`
    #     # `databricks-mixtral-8x7b-instruct`
    #     # `databricks-dbrx-instruct`
    #     "databricks_resources": {
    #         # Databricks Model Serving endpoint name.
    #         "llm_endpoint_name": "databricks-meta-llama-3-70b-instruct",
    #     },
    #     "llm_config": {
    #         # Parameters that control how the LLM responds.
    #         "llm_parameters": {
    #             "enable_safety_filter": True,
    #         },
    #     },
    # },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain code
# MAGIC
# MAGIC Updating your chain code can be a bit more complex because some chain code changes introduce additional chain configuration parameters.  Adding a re-ranker is included as an example chain code experiment.
# MAGIC
# MAGIC To add a chain code experiment:
# MAGIC
# MAGIC 1. Add the chain file to `./chain_code_fixes`
# MAGIC 2. Add the name of the chain file and any required configuration parameters to `CHAIN_CODE_FIXES`. <br/><br/>
# MAGIC
# MAGIC
# MAGIC     ```
# MAGIC     CHAIN_CODE_FIXES = {
# MAGIC         "name_of_experiment": {
# MAGIC             "chain_code_file": "file_name",
# MAGIC             "chain_configuration_override": {
# MAGIC                 # any configuration params required by the china
# MAGIC                 "key": "value"
# MAGIC                 ...
# MAGIC             },
# MAGIC         }
# MAGIC     }
# MAGIC     ```
# MAGIC
# MAGIC **Note: Due to how MLflow logging works, if your chain code requires additional pip packages, they must also be installed in this Notebook.**

# COMMAND ----------

CHAIN_CODE_FIXES = {
    # "reranker": {
    #     # `single_turn_rag_chain_reranker` or `multi_turn_rag_chain_reranker`
    #     "chain_code_file": "single_turn_rag_chain_reranker", 
    #     "chain_configuration_override": {
    #         "retriever_config": {
    #             "parameters": {
    #                 # Number of search results that the retriever returns before re-ranking
    #                 "k": 20
    #             },
    #             "reranker": {
    #                 "k_to_return_after_reranking": 5,
    #                 # Model options from: https://github.com/PrithivirajDamodaran/FlashRank
    #                 # ms-marco-TinyBERT-L-2-v2
    #                 # ms-marco-MiniLM-L-12-v2
    #                 # rank-T5-flan
    #                 # ms-marco-MultiBERT-L-12
    #                 # ce-esci-MiniLM-L12-v2 FT
    #                 "model": "ms-marco-MultiBERT-L-12",
    #             },
    #         }
    #     },
    # }
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Run evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load your evaluation set
# MAGIC
# MAGIC Either load manually as a dictionary or load from a Delta Table.

# COMMAND ----------

# Load from the curated evaluation set's Delta Table

df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# If you do not have an evaluation set, and want to evaluate using a manually curated set of questions, you can use the structure below.

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

# MAGIC %md ## "Compile" each fix into a runnable chain
# MAGIC
# MAGIC Create & run a set of "compiled fixes" to run by merging each fix's config with the baseline configuration.
# MAGIC For example, if you have:
# MAGIC
# MAGIC > `baseline = {'a': {'x': 1, 'y': 2}}`
# MAGIC
# MAGIC > `experiment = {'a': {'x': 4}}`
# MAGIC
# MAGIC The resulting configuration to evaluate will be:
# MAGIC > `strategy = {'a': {'x': 4, 'y': 2}}`

# COMMAND ----------

experiments_to_run = []

for experiment_name, experiment_details in CHAIN_CODE_FIXES.items():
    merged_config = merge_dicts(baseline_chain_config, experiment_details['chain_configuration_override'])
    code_file = experiment_details['chain_code_file']
    
    experiments_to_run.append({
        "experiment_name": experiment_name,
        "chain_config_override": merged_config,
        "code_file": f"chain_code_fixes/{code_file}",
        "data_pipeline_config": baseline_data_pipeline_config

    })

for run_name in DATA_PIPELINE_FIXES_RUN_NAMES:
    # print(experiment)
    pipeline_run = get_mlflow_run(experiment_name=MLFLOW_EXPERIMENT_NAME, run_name=run_name)
    pipeline_chain_config_override = mlflow.artifacts.load_dict(f"{pipeline_run.info.artifact_uri}/chain_config.json")
    data_pipeline_config = mlflow.artifacts.load_dict(f"{pipeline_run.info.artifact_uri}/data_pipeline_config.json")

    # print(pipeline_chain_config_override)
    # print(baseline_chain_config)
    merged_config = merge_dicts(baseline_chain_config, pipeline_chain_config_override)
    experiments_to_run.append({
        "experiment_name": run_name,
        "chain_config_override": merged_config,
        "code_file": baseline_chain_code_file_name,
        "data_pipeline_config": data_pipeline_config
    })

for experiment_name, config_override in CHAIN_CONFIG_FIXES.items():
    merged_config = merge_dicts(baseline_chain_config, config_override)
    experiments_to_run.append({
        "experiment_name": experiment_name,
        "chain_config_override": merged_config,
        "code_file": baseline_chain_code_file_name,
        "data_pipeline_config": baseline_data_pipeline_config

    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation of each "compiled" fix

# COMMAND ----------


for experiment in experiments_to_run:
    code_file = experiment['code_file']
    config_dict = experiment['chain_config_override']
    experiment_name = experiment['experiment_name']
    data_pipeline_config = experiment['data_pipeline_config']
    print(f"Evaluating {experiment_name}...\n----")
    print(f"{config_dict}\n----\n\n")

    with mlflow.start_run(run_name="experiment_"+experiment_name):
        # Tag to differentiate from the data pipeline runs
        mlflow.set_tag("type", "chain")

        # Attach the data pipeline's configuration as parameters
        mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

        # Log to MLflow
        logged_chain_info = mlflow.langchain.log_model(
            lc_model=os.path.join(os.getcwd(), code_file),
            model_config=config_dict,  # Chain configuration
            artifact_path="chain",  # Required
            input_example=config_dict["input_example"],  # Required
            example_no_conversion=True,  # Required to allow the schema to work,
            extra_pip_requirements=["databricks-agents"] # TODO: Remove
        )

        # Evaluate
        eval_results = mlflow.evaluate(
            data=eval_df,
            model=logged_chain_info.model_uri,
            model_type="databricks-agent",
        )

        # Save results for later analysis
        experiment['eval_results'] = eval_results
        experiment['logged_chain_info'] = logged_chain_info

        print("----")


# COMMAND ----------

# MAGIC %md # Decide the best one
# MAGIC
# MAGIC Identify the best chain based on which chain "wins" the most number of times across each `metrics_to_use`.  If multiple chains are equally good, one of them is selected.

# COMMAND ----------

# Select metrics to use for comparison
metrics_to_use = [
    {
        "metric": "response/llm_judged/correctness/rating/percentage",
        "higher_is_better": True,
    },
    {
        "metric": "chain/total_token_count/average",
        "higher_is_better": False,
    },
    {
        "metric": "chain/latency_seconds/average",
        "higher_is_better": False,
    },
]

# Identify the winner for each metric
for metric_info in metrics_to_use:
    metric_name = metric_info['metric']
    baseline = baseline_run.data.metrics[metric_name]
    print(f"Checking for {metric_name}, POC baseline value: {baseline}")
    metric_info['winner'] = 'poc'
    best_score = baseline
    for config in experiments_to_run:
        metric_score = config["eval_results"].metrics[metric_name]
        print(f"   Run {config['experiment_name']}, value: {metric_score}")
        if metric_info['higher_is_better']:
            if metric_score > best_score:
                best_score = metric_score
                metric_info['winner'] = config['experiment_name']
        else:
            if metric_score < best_score:
                best_score = metric_score
                metric_info['winner'] = config['experiment_name']
    print(f"   >> Winner: {metric_info['winner']}")

# Identify the best chain overall
poc_wins = count_wins("poc", metrics_to_use)
for config in experiments_to_run:
    config['wins'] = count_wins(config['experiment_name'], metrics_to_use)
best_tested_config = max(experiments_to_run, key=operator.itemgetter('wins'))
if poc_wins>= best_tested_config['wins']:
    winner = 'poc'
else:
    winner = best_tested_config['experiment_name']
    winning_config = best_tested_config

print(f"Best chain is: {winner}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy the best one
# MAGIC
# MAGIC This step will BOTH:
# MAGIC 1. Deploy the winning chain to the Review App for stakeholder review
# MAGIC 2. Deploy the winning chain as a scalable, production-ready REST API hosted on Mosaic AI Agent Serving
# MAGIC
# MAGIC These steps will only work if the winning model is NOT the POC.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instructions to stakeholders
# MAGIC
# MAGIC Let your stakeholders know how their feedback helped you improve by sharing metrics!
# MAGIC
# MAGIC Make sure you review the instructions to accuratly represent the change in metrics - the example text below assumes accuracy has increased while latency and cost have decreased.
# MAGIC
# MAGIC **Note: If the POC chain is the winning chain, the below cells will result in error messages.**

# COMMAND ----------

# Cost change
after_cost = winning_config['eval_results'].metrics['chain/total_token_count/average']
before_cost = baseline_run.data.metrics['chain/total_token_count/average']
change_in_cost = (after_cost - before_cost) / before_cost

# Latency change
after_latency = winning_config['eval_results'].metrics['chain/latency_seconds/average']
before_latency = baseline_run.data.metrics['chain/latency_seconds/average']
change_in_latency = (after_latency - before_latency) / before_latency

# Reviewer message
instructions_to_reviewer = f"""## {RAG_APP_NAME} v2

**Thank you for your invaluable feedback on our POC. Your feedback has allowed us to:**
- **Increase** the app's overall accuracy from {round(baseline_run.data.metrics['response/llm_judged/correctness/rating/percentage'])*100}% to **{round(winning_config['eval_results'].metrics['response/llm_judged/correctness/rating/percentage'])*100}%**
- **Increase** the app's speed by **{round(abs(change_in_latency*100))}%**
- **Decrease** our cost of running the app by **{round(abs(change_in_cost*100))}%**

### Please test the v2 of the application following the same instructions as before:

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

# MAGIC %md
# MAGIC ## Deploy the best model
# MAGIC
# MAGIC Deploys to:
# MAGIC - Review App
# MAGIC - Scalable, production-ready REST API

# COMMAND ----------

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

if winner != 'poc':
    # Register the winning to the UC model
    uc_registered_model_info = mlflow.register_model(model_uri=winning_config['logged_chain_info'].model_uri, name=UC_MODEL_NAME)

    # Deploy to enable the Review APP and create an API endpoint
    deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

    browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
    print(f"View deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

    # Add the user-facing instructions to the Review App
    agents.set_review_instructions(UC_MODEL_NAME, instructions_to_reviewer)

    # Wait for the Review App to be ready
    while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
        print("Waiting for endpoint to deploy.  This can take 15 - 20 minutes.  Waiting for 5 minutes before checking again...")
        time.sleep(60*5)

    print(f"Review App: {deployment_info.review_app_url}")
