# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 10 minute demo of Mosaic AI Agent Framework & Agent Evaluation
# MAGIC
# MAGIC #### TLDR; this notebook will:
# MAGIC 1. Deploy a RAG application built with [Agent Framework](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) to the [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) review application
# MAGIC     - The review application is used by your business stakeholders to provide feedback on your app
# MAGIC 2. Evaluate the quality of the application with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) and MLflow
# MAGIC     - These AI-assisted evaluations are used by developers to improve the application's quality
# MAGIC
# MAGIC #### Products used:
# MAGIC - [**Mosaic AI Agent Framework**](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) SDK to quickly and safely build high-quality RAG applications.
# MAGIC - [**Mosaic AI Agent Evaluation**](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) AI-assisted evaluation tool to determines if outputs are high-quality.  Provides an intuitive UI to get feedback from human stakeholders.
# MAGIC - [**Mosaic AI Model Serving**](https://docs.databricks.com/generative-ai/deploy-agent.html) Hosts the application's logic as a production-ready, scalable REST API.
# MAGIC - [**MLflow**](https://docs.databricks.com/mlflow/index.html) Tracks and manages the application lifecycle, including evaluation results and application code/config
# MAGIC - [**Generative AI Cookbook**](https://ai-cookbook.io/) A definitive how-to guide, backed by a code repo, for building high-quality Gen AI apps, developed in partnership with Mosaic AI’s research team.
# MAGIC
# MAGIC
# MAGIC #### Agent Evaluation review application
# MAGIC <img src="https://ai-cookbook.io/_images/review_app2.gif" style="float: left;  margin-left: 10px" width="60%">
# MAGIC
# MAGIC #### Agent Evaluation outputs in MLflow
# MAGIC <img src="https://ai-cookbook.io/_images/mlflow-eval-agent.gif" style="float: left;  margin-left: 10px" width="60%">
# MAGIC
# MAGIC #### Generative AI Cookbook
# MAGIC <img src="https://raw.githubusercontent.com/databricks/genai-cookbook/e280f9a33137682cf90868a3ac95a775e4f958ef/genai_cookbook/images/5-hands-on/cookbook.png" style="float: left;  margin-left: 10px" width="60%">

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch databricks-sdk langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Load the necessary data and code from the Databricks Cookbook repo
# MAGIC
# MAGIC The following cell clones the Generative AI cookbook repo from `https://github.com/databricks/genai-cookbook` into a folder `genai-cookbook` in the same folder as this notebook using a [Git Folder](https://docs.databricks.com/en/repos/index.html).  
# MAGIC
# MAGIC Alternatively, you can manually clone the Git repo `https://github.com/databricks/genai-cookbook` to a folder `genai-cookbook`.

# COMMAND ----------

import os
from databricks.sdk.core import DatabricksError
from databricks.sdk import WorkspaceClient

CURRENT_FOLDER = os.getcwd()
QUICK_START_REPO_URL = "https://github.com/epec254/staging-cookbook.git"
QUICK_START_REPO_URL = "https://github.com/databricks/genai-cookbook.git"
QUICK_START_REPO_SAVE_FOLDER = "genai-cookbook"

if os.path.isdir(QUICK_START_REPO_SAVE_FOLDER):
    raise Exception(
        f"{QUICK_START_REPO_SAVE_FOLDER} folder already exists, please change the variable QUICK_START_REPO_SAVE_FOLDER to be a non-existant path."
    )

# Clone the repo
w = WorkspaceClient()
try:
    w.repos.create(
        url=QUICK_START_REPO_URL, provider="github", path=f"{CURRENT_FOLDER}/{QUICK_START_REPO_SAVE_FOLDER}"
    )
    print(f"Cloned sample code repo to: {QUICK_START_REPO_SAVE_FOLDER}")
except DatabricksError as e:
    if e.error_code == "RESOURCE_ALREADY_EXISTS":
        print("Repo already exists. Skipping creation")
    else:
        raise Exception(
            f"Failed to clone the quick start code.  You can manually import this by creating a Git folder from the contents of {QUICK_START_REPO_URL} in the {QUICK_START_REPO_SAVE_FOLDER} folder in your workspace and then re-running this Notebook."
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application configuration
# MAGIC
# MAGIC We've selected defaults for the following parameters based on your user name, but inspect and change if you prefer to use existing resources.  Any missing resources will be created in the next step.
# MAGIC
# MAGIC 1. `UC_CATALOG` & `UC_SCHEMA`: [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored
# MAGIC 2. `UC_MODEL_NAME`: Unity Catalog location to log and store the chain's model
# MAGIC 3. `VECTOR_SEARCH_ENDPOINT`: [Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to host the resulting vector index

# COMMAND ----------

# Use the current user name to create any necesary resources
w = WorkspaceClient()
user_name = w.current_user.me().user_name.split("@")[0].replace(".", "")

# UC Catalog & Schema where outputs tables/indexs are saved
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f'{user_name}_catalog'
UC_CATALOG = "rag_m1_release"
UC_SCHEMA = f'rag_{user_name}'
UC_SCHEMA = "10_mins"

# UC Model name where the POC chain is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{user_name}_agent_quick_start"

# Vector Search endpoint where index is loaded
# If this does not exist, it will be created
VECTOR_SEARCH_ENDPOINT = f'{user_name}_vector_search'

# COMMAND ----------

# MAGIC %md ## Check if the UC Catalog & Vector Search endpoint exist; create otherwise

# COMMAND ----------

# MAGIC %run ./genai-cookbook/quick_start_demo/helper_create_catalog_and_vector_index

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Build & deploy the application
# MAGIC
# MAGIC Below is a high-level overview of the architecture we will deploy:
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic.png?raw=true" style="width: 800px; margin-left: 10px">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-3.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC ## 1/ Create the Vector Search Index
# MAGIC
# MAGIC First, we copy the sample data to a Delta Table and sync to a Vector Search index.  Here, we use the [gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) embedding model hosted on [Databricks Foundational Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html).

# COMMAND ----------

# UC locations to store the chunked documents & index
CHUNKS_DELTA_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked2"
CHUNKS_VECTOR_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.databricks_docs_chunked_index2"

# COMMAND ----------

from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient

# Workspace URL for printing links to the delta table/vector index
workspace_url = SparkSession.getActiveSession().conf.get(
    "spark.databricks.workspaceUrl", None
)

# Vector Search client
vsc = VectorSearchClient(disable_notice=True)

# Load the chunked data to Delta Table & enable change-data capture to allow the table to sync to Vector Search
chunked_docs_df = spark.read.parquet(
    f"file:{QUICK_START_REPO_SAVE_FOLDER}/quick_start_demo/chunked_databricks_docs.snappy.parquet"
)
chunked_docs_df.write.format("delta").saveAsTable(CHUNKS_DELTA_TABLE)
spark.sql(
    f"ALTER TABLE {CHUNKS_DELTA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(
    f"View Delta Table at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_DELTA_TABLE.split('.')[-1]}"
)

# Embed and sync chunks to a vector index
print(
    f"Embedding docs & creating Vector Search Index, this will take ~5 - 10 minutes.\nView Index Status at: https://{workspace_url}/explore/data/{UC_CATALOG}/{UC_SCHEMA}/{CHUNKS_VECTOR_INDEX.split('.')[-1]}"
)

index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=CHUNKS_VECTOR_INDEX,
    primary_key="chunk_id",
    source_table_name=CHUNKS_DELTA_TABLE,
    pipeline_type="TRIGGERED",
    embedding_source_column="chunked_text",
    embedding_model_endpoint_name="databricks-gte-large-en",
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2/ Deploy to the review application
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-chain-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Now that our Vector Search index is ready, let's prepare the RAG chain and deploy it to the review application backed by a scalable-production ready REST API on Model serving.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1/ Configuring our Chain parameters
# MAGIC
# MAGIC Databricks makes it easy to parameterize your chain with MLflow Model Configurations. Later, you can tune application quality by adjusting these parameters, such as the system prompt or retrieval settings.  Most applications will include many more parameters, but for this demo, we'll keep the configuration to a minimum.

# COMMAND ----------

chain_config = {
    "llm_model_serving_endpoint_name": "databricks-dbrx-instruct",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,  # Endoint for vector search
    "vector_search_index": f"{UC_CATALOG}.{UC_SCHEMA}.{CHUNKS_VECTOR_INDEX}",
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""", # LLM Prompt template
}

# Here, we define an input example in the schema required by Agent Framework
input_example = {"messages": [ {"role": "user", "content": "What is Retrieval-augmented Generation?"}]}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 2.1/ Log the application & view trace
# MAGIC
# MAGIC We first register the chain as an MLflow model and inspect the MLflow Trace to understand what is happening inside the chain.
# MAGIC
# MAGIC #### MLflow trace
# MAGIC <br/>
# MAGIC <img src="https://ai-cookbook.io/_images/mlflow_trace2.gif" width="80%" style="margin-left: 10px">
# MAGIC

# COMMAND ----------

import mlflow

# Log the model to MLflow
with mlflow.start_run(run_name="databricks-docs-bot"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            f"{QUICK_START_REPO_SAVE_FOLDER}/quick_start_demo/sample_rag_chain",
        ),  # Chain code file from the quick start repo
        model_config=chain_config,  # Chain configuration set above
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
    )

# Test the chain locally to see the MLflow Trace
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 2.1/ Deploy the application
# MAGIC
# MAGIC Now, we:
# MAGIC 1. Register the application in Unity Catalog
# MAGIC 2. Use Agent Framework to deploy to the Quality Lab review application
# MAGIC
# MAGIC Along side the review ap, a scalable, production-ready Model Serving endpoint is also deployed.
# MAGIC
# MAGIC #### Agent Evaluation review application
# MAGIC <img src="https://ai-cookbook.io/_images/review_app2.gif" width="90%">

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3/ Use Agent Evaluation to evaluate your application
# MAGIC
# MAGIC ## 3.1/ Have stakeholders chat your bot to build your evaluation dataset
# MAGIC
# MAGIC Normally, you would now give access to internal domain experts and have them test and review the bot.  **Your domain experts do NOT need to have Databricks Workspace access** - you can assign permissions to any user in your SSO if you have enabled [SCIM](https://docs.databricks.com/en/admin/users-groups/scim/index.html)
# MAGIC
# MAGIC This is a critical step to build or improve your evaluation dataset: have users ask questions to your bot, and provide the bot with output answer when they don't answer properly.
# MAGIC
# MAGIC Your applicaation is automatically capturing all stakeholder questions and bot responses, including the MLflow Trace for each, into Delta Tables in your Lakehouse. On top of that, Databricks makes it easy to track feedback from your end user: if the chatbot doesn't give a good answer and the user gives a thumbdown, their feedback is included in the Delta Tables.
# MAGIC
# MAGIC Your evaluation dataset forms the basis of your development workflow to improve quality: identifying the root causes of quality issues and then objectively measuring the impact of your fixes.
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://ai-cookbook.io/_images/review_app2.gif" style="float: left;  margin-left: 10px" width="80%">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2/ Run Evaluation of your Chain
# MAGIC
# MAGIC Now, let's use everage Agent Evaluation's specialized AI evaluators to evaluate our model performance.  Agent Evaluation is integrated into `mlflow.evaluate(...)`, all you need to do is pass `model_type="databricks-agent"`.
# MAGIC
# MAGIC For this demo, we use a toy 10 question evaluation dataset.  Read more about our [best practices](https://ai-cookbook.io/nbs/4-evaluation-eval-sets.html) on the size of your evaluation dataset.
# MAGIC
# MAGIC <img src="https://ai-cookbook.io/_images/mlflow-eval-agent.gif" style="float: left;  margin-left: 10px" width="80%">

# COMMAND ----------

import pandas as pd

sample_eval_set = [
    {
        "request_id": "5482",
        "request": "What happens if I try to access an index that is out of bounds in an array using the [ ] operator in Databricks SQL when spark.sql.ansi.enabled is set to false?",
        "response": "If you try to access an index that is out of bounds in an array using the [ ] operator in Databricks SQL when `spark.sql.ansi.enabled` is set to false, Databricks will return `NULL` instead of raising an `INVALID_ARRAY_INDEX` error. However, if `spark.sql.ansi.enabled` is set to true, an `INVALID_ARRAY_INDEX` error will be raised.",
        "expected_retrieved_context": [
            {
                "content": "[ ] (bracket sign) operator  \nApplies to: Databricks SQL Databricks Runtime  \nReturns an array element or map value given an index or key.  \nSyntax\nSyntax\nexpr [ keyExpr ]  \nNote that the use of brackets here is as literal text and not indicating optional syntax.\n\nArguments\nArguments\nexpr: An ARRAY or MAP expression.  \nkeyExpr: If expr is an ARRAY an integral numeric. Otherwise an expression matching the type of the key of the MAP.\n\nReturns\nReturns\nThe result type is the element type of the ARRAY or the value type of the MAP.  \nThe first element of an ARRAY is at index 0.  \nIf the keyExpr is not a valid key for the MAP expr Databricks returns null.  \nIf the keyExpr is out of bound for the ARRAY expr Databricks raises a INVALID_ARRAY_INDEXerror.  \nNote  \nIn Databricks Runtime, if spark.sql.ansi.enabled is false, the operator returns NULL instead of an out of bounds error.\n\nExamples\nExamples\n> SELECT a[2] FROM VALUES(array(10, 20, 30)) AS T(a); 30 > SELECT m[1] FROM VALUES(map(1, 'Hello', 2, 'World')) AS T(m); Hello\n\nRelated functions\nRelated functions\narray function  \narray_contains function  \narray_position function  \nmap function  \nmap_keys function  \nmap_values function",
                "doc_uri": "https://docs.databricks.com/sql/language-manual/functions/bracketsign.html",
            }
        ],
        "expected_response": "If you try to access an index that is out of bounds in an array using the [ ] operator in Databricks SQL when spark.sql.ansi.enabled is set to false, Databricks will return NULL instead of raising an error.",
    },
    {
        "request_id": "2112",
        "request": "Why is a long-running stage in my Spark job only showing one task, and how can I resolve this issue?",
        "response": "A long-running stage in a Spark job with only one task could be caused by expensive UDFs on small data, window functions without a `PARTITION BY` statement, reading from an unsplittable file type, setting the `multiLine` option when reading a JSON or CSV file, schema inference of a large file, or using `repartition(1)` or `coalesce(1)`. To diagnose this issue, you can check the Spark UI to identify the longest stage of the job and look at the Input, Output, Shuffle Read, and Shuffle Write columns. If there is only one task, it could be a sign of a problem. To resolve this issue, you can try optimizing your Spark job by avoiding the aforementioned causes. For example, you can avoid using `repartition(1)` or `coalesce(1)` unless necessary, use a splittable file type, and optimize your UDFs and window functions.",
        "expected_retrieved_context": [
            {
                "content": "One Spark task  \nIf you see a long-running stage with just one task, that’s likely a sign of a problem. While this one task is running only one CPU is utilized and the rest of the cluster may be idle. This happens most frequently in the following situations:  \nExpensive UDF on small data  \nWindow function without PARTITION BY statement  \nReading from an unsplittable file type. This means the file cannot be read in multiple parts, so you end up with one big task. Gzip is an example of an unsplittable file type.  \nSetting the multiLine option when reading a JSON or CSV file  \nSchema inference of a large file  \nUse of repartition(1) or coalesce(1)",
                "doc_uri": "https://docs.databricks.com/optimizations/spark-ui-guide/one-spark-task.html",
            }
        ],
        "expected_response": "A long-running stage with one task in a Spark job could be due to several reasons such as:\n\n1. Expensive User Defined Functions (UDFs) on small data\n2. Window function without a PARTITION BY statement\n3. Reading from an unsplittable file type like gzip\n4. Setting the multiLine option when reading a JSON or CSV file\n5. Schema inference of a large file\n6. Use of repartition(1) or coalesce(1)\n\nTo resolve this issue, you can:\n\n1. Optimize your UDFs or replace them with built-in functions if possible.\n2. Ensure that you have a proper PARTITION BY statement in your window functions.\n3. Avoid using unsplittable file types like gzip. Instead, use splittable file types like snappy or lz4.\n4. Avoid setting the multiLine option when reading JSON or CSV files.\n5. Perform schema inference on a small sample of your data and then apply it to the entire dataset.\n6. Avoid using repartition(1) or coalesce(1) unless necessary.\n\nBy implementing these changes, you should be able to resolve the issue of a long-running stage with only one task in your Spark job.",
    },
    {
        "request_id": "5054",
        "request": "How can I represent 4-byte single-precision floating point numbers in Databricks SQL and what are their limits?",
        "response": "4-byte single-precision floating point numbers in Databricks SQL are represented using the `FLOAT` data type. The range of numbers that can be represented is from -3.402E+38 to +3.402E+38, including negative infinity, positive infinity, and NaN (not a number). The literals for `FLOAT` type can be expressed in base-10 or base-2 format, with the `F` postfix and `E` exponent being case insensitive. However, due to the base-2 representation, the literal may not be exact. For accurate representation of fractional or large base-10 numbers, it is recommended to use the `DECIMAL` data type.",
        "expected_retrieved_context": [
            {
                "content": "FLOAT type  \nApplies to: Databricks SQL Databricks Runtime  \nRepresents 4-byte single-precision floating point numbers.  \nSyntax\nSyntax\n{ FLOAT | REAL }\n\nLimits\nLimits\nThe range of numbers is:  \n-∞ (negative infinity)  \n-3.402E+38 to -1.175E-37  \n0  \n+1.175E-37 to +3.402E+38  \n+∞ (positive infinity)  \nNaN (not a number)\n\nLiterals\nLiterals\ndecimal_digits [ exponent ] F | [ + | - ] digit [ ... ] [ exponent ] F decimal_digits: [ + | - ] { digit [ ... ] . [ digit [ ... ] ] | . digit [ ... ] } exponent: E [ + | - ] digit [ ... ]  \ndigit: Any numeral from 0 to 9.  \nThe F postfix and E exponent are case insensitive.\n\nNotes\nNotes\nFLOAT is a base-2 numeric type. When given a literal which is base-10 the representation may not be exact. Use DECIMAL type to accurately represent fractional or large base-10 numbers.\n\nExamples\nExamples\n> SELECT +1F; 1.0 > SELECT 5E10F; 5E10 > SELECT 5.3E10F; 5.3E10 > SELECT -.1F; -0.1 > SELECT 2.F; 2.0 > SELECT -5555555555555555.1F -5.5555558E15 > SELECT CAST(6.1 AS FLOAT) 6.1\n\nRelated\nRelated\nTINYINT type  \nSMALLINT type  \nINT type  \nBIGINT type  \nDECIMAL type  \nDOUBLE type  \ncast function  \nSpecial floating point values",
                "doc_uri": "https://docs.databricks.com/sql/language-manual/data-types/float-type.html",
            }
        ],
        "expected_response": "4-byte single-precision floating point numbers can be represented in Databricks SQL using the `FLOAT` or `REAL` syntax. The range of numbers that can be represented is from -3.402E+38 to +3.402E+38, including negative infinity, positive infinity, and NaN (not a number). Here are some examples of how to represent these numbers:\n\n* `+1F` represents 1.0\n* `5E10F` represents 5E10\n* `5.3E10F` represents 5.3E10\n* `-.1F` represents -0.1\n* `2.F` represents 2.0\n* `-5555555555555555.1F` represents -5.5555558E15\n* `CAST(6.1 AS FLOAT)` represents 6.1\n\nNote that `FLOAT` is a base-2 numeric type, so the representation of base-10 literals may not be exact. If you need to accurately represent fractional or large base-10 numbers, consider using the `DECIMAL` type instead.",
    },
    {
        "request_id": "2003",
        "request": "How can I identify the reason for failing executors in my Databricks workspace, and what steps can I take to resolve memory issues?",
        "response": "To identify the reason for failing executors in your Databricks workspace, you should first check the compute's Event log in the Spark UI to see if there's any explanation for why the executors failed. If you don't find any information in the event log, navigate to the Executors tab in the Spark UI to get the logs from the failed executors.\n\nThe most common reasons for executors being removed are autoscaling, spot instance losses, and executors running out of memory. If you see any failing jobs, click on them to get to their pages, scroll down to see the failed stage and a failure reason, and check the failed tasks to identify the issue.\n\nIf you suspect a memory issue, you can verify it by doubling the memory per core to see if it impacts your problem. If it takes longer to fail with the extra memory or doesn't fail at all, that's a good sign that you're on the right track. If you can fix your issue by increasing the memory, great! If it doesn't fix the issue, or you can't bear the extra cost, you should dig deeper into memory issues.",
        "expected_retrieved_context": [
            {
                "content": "Failing jobs or executors removed  \nSo you’re seeing failed jobs or removed executors:  \nThe most common reasons for executors being removed are:  \nAutoscaling: In this case it’s expected and not an error. See Enable autoscaling.  \nSpot instance losses: The cloud provider is reclaiming your VMs. You can learn more about Spot instances here.  \nExecutors running out of memory  \nFailing jobs\nFailing jobs\nIf you see any failing jobs click on them to get to their pages. Then scroll down to see the failed stage and a failure reason:  \nYou may get a generic error. Click on the link in the description to see if you can get more info:  \nIf you scroll down in this page, you will be able to see why each task failed. In this case it’s becoming clear there’s a memory issue:\n\nFailing executors\nFailing executors\nTo find out why your executors are failing, you’ll first want to check the compute’s Event log to see if there’s any explanation for why the executors failed. For example, it’s possible you’re using spot instances and the cloud provider is taking them back.  \nSee if there are any events explaining the loss of executors. For example you may see messages indicating that the cluster is resizing or spot instances are being lost.  \nIf you are using spot instances, see Losing spot instances.  \nIf your compute was resized with autoscaling, it’s expected and not an error. See Learn more about cluster resizing.  \nIf you don’t see any information in the event log, navigate back to the Spark UI then click the Executors tab:  \nHere you can get the logs from the failed executors:\n\nNext step\nNext step\nIf you’ve gotten this far, the likeliest explanation is a memory issue. The next step is to dig into memory issues. See Spark memory issues.",
                "doc_uri": "https://docs.databricks.com/optimizations/spark-ui-guide/failing-spark-jobs.html",
            }
        ],
        "expected_response": "1. Identify failing executors: In your Databricks workspace, navigate to the compute's Event log to check for any explanations regarding executor failures. Look for messages indicating spot instance losses or cluster resizing due to autoscaling. If using spot instances, refer to 'Losing spot instances' documentation. For autoscaling, refer to 'Learn more about cluster resizing' documentation.\n\n2. Check executor logs: If no information is found in the event log, go to the Spark UI and click the Executors tab. Here, you can access logs from failed executors to investigate further.\n\n3. Identify memory issues: If the above steps do not provide a clear reason for failing executors, it is likely a memory issue. To dig into memory issues, refer to the 'Spark memory issues' documentation.\n\n4. Resolve memory issues: To resolve memory issues, consider the following steps:\n\n   a. Increase executor memory: Allocate more memory to executors by adjusting the 'spark.executor.memory' property in your Spark configuration.\n\n   b. Increase driver memory: Allocate more memory to the driver by adjusting the 'spark.driver.memory' property in your Spark configuration.\n\n   c. Use off-heap memory: Enable off-heap memory by setting the 'spark.memory.offHeap.enabled' property to 'true' and allocating off-heap memory using the 'spark.memory.offHeap.size' property.\n\n   d. Optimize data processing: Review your data processing workflows and optimize them for memory efficiency. This may include reducing data shuffling, using broadcast variables, or caching data strategically.\n\n   e. Monitor memory usage: Monitor memory usage in your Databricks workspace to identify potential memory leaks or inefficient memory utilization. Use tools like the Spark UI, Ganglia, or Grafana to monitor memory usage.",
    },
]

eval_df = pd.DataFrame(sample_eval_set)
display(eval_df)

# COMMAND ----------

with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df, # Your evaluation set
        model=logged_chain_info.model_uri, # previously logged model
        model_type="databricks-agent", # activate Mosaic AI Agent Evaluation
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # What's next?
# MAGIC
# MAGIC
# MAGIC ## Code-based quick starts
# MAGIC
# MAGIC | Time required | Outcome | Link |
# MAGIC |------ | ---- | ---- |
# MAGIC | 🕧 <br/> 10 minutes | Sample RAG app deployed to web-based chat app that collects feedback | ✅ |
# MAGIC | 🕧🕧🕧 <br/>60 minutes | POC RAG app with *your data* deployed to a chat UI that can collect feedback from your business stakeholders | [Deploy POC w/ your data](https://ai-cookbook.io/nbs/5-hands-on-build-poc.html)|
# MAGIC | 🕧🕧 <br/>30 minutes | Comprehensive quality/cost/latency evaluation of your POC app | - [Evaluate your POC](https://ai-cookbook.io/nbs/5-hands-on-evaluate-poc.html) <br/> - [Identify the root causes of quality issues](https://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-1.html) |
# MAGIC
# MAGIC ## Browse the code samples
# MAGIC
# MAGIC Open the `./genai-cookbook/rag_app_sample_code` folder that was synced to your Workspace by this notebook.  Documentation [here](https://ai-cookbook.io/nbs/6-implement-overview.html).
# MAGIC
# MAGIC ## Read the [Generative AI Cookbook](https://ai-cookbook.io)!
# MAGIC
# MAGIC **TLDR;** the [cookbook]((https://ai-cookbook.io) and its sample code will take you from initial POC to high-quality production-ready application using [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) and [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) on the Databricks platform.
# MAGIC
# MAGIC The Databricks Generative AI Cookbook is a definitive how-to guide for building *high-quality* generative AI applications. *High-quality* applications are applications that:
# MAGIC 1. **Accurate:** provide correct responses
# MAGIC 2. **Safe:** do not deliver harmful or insecure responses
# MAGIC 3. **Governed:** respect data permissions & access controls and track lineage
# MAGIC
# MAGIC Developed in partnership with Mosaic AI's research team, this cookbook lays out Databricks best-practice development workflow for building high-quality RAG apps: *evaluation driven development.* It outlines the most relevant knobs & approaches that can increase RAG application quality and provides a comprehensive repository of sample code implementing those techniques. 
# MAGIC
# MAGIC <br/>
# MAGIC
# MAGIC <img src="https://ai-cookbook.io/_images/dbxquality.png" style="margin-left: 10px" width="80%">
