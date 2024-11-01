# Databricks notebook source
# MAGIC %md
# MAGIC # Agent
# MAGIC
# MAGIC Use this notebook to iterate on the code and configuration of your Agent.
# MAGIC
# MAGIC By the end of this notebook, you will have 1+ registered versions of your Agent, each coupled with a detailed quality evaluation.  To interact with this Agent through the Playground or share with your business stakeholders for feedback, use the following notebooks.
# MAGIC
# MAGIC For each version, you will have an MLflow run inside your MLflow experiment that contains:
# MAGIC - Your Agent's code & config
# MAGIC - Evaluation metrics for cost, quality, and latency

# COMMAND ----------

# MAGIC %md
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC We suggest the following approach to using this notebook to build and iterate on your Agent's quality.  
# MAGIC 1. Build an initial version of your Agent by tweaking the smart default settings and code in this notebook.
# MAGIC
# MAGIC 2. Vibe check & iterate on the Agent's quality to reach a "not embarassingly bad" level of quality.  Test 5 - 10 questions using MLflow Tracing & Agent Evaluation's quality root cause analysis to guide your iteration.
# MAGIC
# MAGIC 3. Use the later notebooks to share your Agent with stakeholders to collect feedback that you will turn into an evaluation set with questions/correct responses labeled by your stakeholders.
# MAGIC
# MAGIC 4. Use this notebook to Agent Evaluation using this evaluation set.
# MAGIC
# MAGIC 5. Same as step 2, use MLflow Tracing & Agent Evaluation's quality root cause analysis to guide your iteration.  Iteratively try and evaluate various strategies to improve the quality of your agent and/or retriever.  For a deep dive on these strategies, view AI cookbook's [retrieval](https://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-1-retrieval.html) and [generation](https://ai-cookbook.io/nbs/5-hands-on-improve-quality-step-1-generation.html) guides.
# MAGIC
# MAGIC 6. Repeat step 3 to collect more feedback, then repeat steps 4 and 5 to further improve quality

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

# Shared imports
from datetime import datetime
from IPython.display import display_markdown

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0️⃣ Setup: Load the Agent's configuration that is shared with the other notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC #### 🚫✏️ (Optional) Change Agent's shared storage location configuration
# MAGIC
# MAGIC ** If you configured `00_shared_config`, just run this cell as-is.**
# MAGIC
# MAGIC From the shared configuration, this notebook uses:
# MAGIC * The Evaluation Set stored in Unity Catalog
# MAGIC * The MLflow experiment for tracking Agent verions & their quality evaluations
# MAGIC
# MAGIC *These values can be set here if you want to use this notebook independently.*

# COMMAND ----------

from utils.cookbook.agent_config import CookbookAgentConfig
import mlflow

# Load the shared configuration
cookbook_shared_config = CookbookAgentConfig.from_yaml_file('./configs/cookbook_config.yaml')

# Print configuration 
cookbook_shared_config.pretty_print()

# Set the MLflow Experiment that is used to track metadata about each run of this Data Pipeline.
experiment_info = mlflow.set_experiment(cookbook_shared_config.mlflow_experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1️⃣ Iterate on the Agent's code & config to improve quality
# MAGIC
# MAGIC The below cells are used to execute your inner dev loop to improve the Agent's quality.
# MAGIC
# MAGIC If you are creating this Agent for the first time, you will:
# MAGIC 1. Review the smart defaults provided in the Agent's code & configuration
# MAGIC 2. Vibe check the Agent for 1 query to verify it works
# MAGIC
# MAGIC We suggest the following inner dev loop:
# MAGIC 1. Run the Agent for 1+ queries or your evaluation set
# MAGIC 2. Determine if the Agent's output is correct for those queries e.g., high quality
# MAGIC 3. Based on that assessment, make changes to the code/config to improve quality
# MAGIC 4. 🔁 Re-run the Agent for the same queries, repeating this cycle.
# MAGIC 5. Once you have a version of the Agent with sufficient quality, log the Agent to MLflow
# MAGIC 6. Use the next notebooks to share the Agent with your stakeholders & collect feedback
# MAGIC 7. Add stakeholder's queries & feedback to your evaluation set
# MAGIC 8. 🔁 Use that evaluation set to repeat this cycle

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### ✅✏️ Change the Agent's code & config

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ ⚙️ Adjust the Agent's configuration
# MAGIC
# MAGIC Here, we use the MLflow [ModelConfig](https://docs.databricks.com/en/generative-ai/create-log-agent.html#use-parameters-to-configure-the-agent) primitive to parameterize your Agent's code with common settings you will tune to improve quality, such as prompts.
# MAGIC
# MAGIC > *Note: Our template Agents use [Pydantic](https://docs.pydantic.dev/latest/) models, which are thin wrappers around Python dictionaries.  Pydantic allows us to define the initial parameters Databricks suggests for tuning quality and allows this notebook to validate parameters changes you make.*
# MAGIC > 
# MAGIC > *If you prefer, you can switch to using a native Python dictionary for parameterization.  Since MLflow ModelConfig only accepts YAML files or dictionaries, we dump the Pydantic model to a YAML file before passing it to MLflow ModelConfig.*
# MAGIC
# MAGIC We use Pydantic to define tools and support automatically serializing their classnames and configs to YAML that can be
# MAGIC loaded back.
# MAGIC
# MAGIC You can (and often will need to) add or adjust the parameters in our template.  To add/modify/delete a parameter, you can either:
# MAGIC 1. Modify the Pydantic classes in `utils.agents.config`
# MAGIC 2. Create a Python dictionary in this notebook to replace the Pydantic class

# COMMAND ----------

# Import Pydantic models
from utils.agents.config import (
    AgentConfig,
)
from utils.agents.llm import LLMConfig, LLMParametersConfig
from utils.agents.vector_search import VectorSearchRetriever, VectorSearchRetrieverTool, VectorSearchRetrieverConfig, VectorSearchRetrieverInputSchema, RetrieverOutputSchema
from utils.agents import get_agent_dependencies, log_pyfunc_agent
import json
import yaml

# # View Retriever config documentation by inspecting the docstrings
#
# help(VectorSearchRetrieverConfig)
# help(RetrieverOutputSchema)
#
# # View documentation for the parameters by inspecting the docstring
#
# help(LLMConfig)
# help(LLMParametersConfig)
# help(AgentConfig)

# COMMAND ----------

########################
# #### 🚫✏️ Load the Vector Index location from the data pipeline configuration
########################

# This loads the Vector Index Unity Catalog location from the data pipeline configuration.

# Usage:
# - If you used `01_data_pipeline` to create your Vector Index, run this cell.
# - If your Vector Index was created elsewhere, skip this cell and set the UC location in the Retriever config.

from utils.data_pipeline.data_pipeline_config import UnstructuredDataPipelineStorageConfig

datapipeline_output_config = UnstructuredDataPipelineStorageConfig.from_yaml_file('./configs/data_pipeline_storage_config.yaml')


########################
# #### ✅✏️ Retriever tool that connects to the Vector Search index
########################

retriever_config = VectorSearchRetrieverConfig(
    vector_search_index=datapipeline_output_config.vector_index,  # UC Vector Search index
    # Retriever schema, this is required by Agent Evaluation to:
    # 1. Enable the Review App to properly display retrieved chunks
    # 2. Enable metrics / LLM judges to understand which fields to use to measure the retriever
    # Each is a column name within the `vector_search_index`
    vector_search_schema=RetrieverOutputSchema(
        primary_key="chunk_id",  # The column name in the retriever's response referred to the unique key
        chunk_text="content_chunked",  # The column name in the retriever's response that contains the returned chunk
        document_uri="doc_uri",  # The URI of the chunk - displayed as the document ID in the Review App
        additional_metadata_columns=[],  # Additional columns to return from the vector database and present to the LLM
    ),
    # Parameters defined by Vector Search docs: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#query-a-vector-search-endpoint
    vector_search_parameters=VectorSearchRetrieverInputSchema(
        num_results=5,  # Number of search results that the retriever returns
        query_type="ann",  # Type of search: ann or hybrid
    ),
    vector_search_threshold=0.0,  # 0 to 1, similarity threshold cut off for retrieved docs.  Increase if the retriever is returning irrelevant content.
    chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",  # Prompt template used to format the retrieved information into {context} in `prompt_template`
    prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",  # Prompt used to present the retrieved information to the LLM
)

########################
#### ✅✏️ LLM configuration
########################

llm_config = LLMConfig(
    llm_endpoint_name="databricks-meta-llama-3-1-405b-instruct",  # Model serving endpoint
    llm_system_prompt_template=(
        """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the outputs from tools.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
    ),  # System prompt template
    llm_parameters=LLMParametersConfig(
        temperature=0.01, max_tokens=1500
    ),  # LLM parameters
)

agent_config = AgentConfig(
    llm_config=llm_config,
    tools=[VectorSearchRetrieverTool(
        vector_search_retriever=VectorSearchRetriever(retriever_config),
        # the prompt used to describe when the tool so the LLM can decide when it is relevant to call.
        tool_description_prompt="Search for documents that are relevant to a user's query about the [REPLACE WITH DESCRIPTION OF YOUR DOCS].",
        # the prompt that describes the tool's name.  Used in combination with `tool_description_prompt` to describe when the tool so the LLM can decide when it is relevant to call.
        tool_name="retrieve_documents",
        # Retriever prompts: Tune these prompts if the Agent uses the retriever incorrectly e.g., doesn't call the retriever tool for the right queries or translates the user's intent to a query incorrectly.
        retriever_query_parameter_prompt="The query to find documents for.", # the prompt used to describe what inputs should go in the 'query' parameter which is used by the vector index to search for relevant documents
    )],
    input_example={
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
)


########################
##### 🚫✏️ Dump the configuration to a YAML
########################

# We dump the Pydantic model to a YAML file because:
# 1. MLflow ModelConfig only accepts YAML files or dictionaries
# 2. When importing the Agent's code, it needs to read this configuration
def write_dict_to_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

with open("./configs/agent_model_config.yaml", "w") as handle:
    agent_config_yml = agent_config.to_yaml()
    handle.write(agent_config_yml)

########################
#### Print resulting config to the console
########################
print(json.dumps(agent_config.model_dump(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Adjust the Agent's code
# MAGIC
# MAGIC Here, we import the Agent's code so we can run the Agent locally within the notebook.  To modify the code, open this Notebook in a separate window, make your changes, and re-run this cell.
# MAGIC
# MAGIC **Important: Typically, when building the first version of your agent, you will not need to modify the code.**

# COMMAND ----------

# MAGIC %run ./agents/function_calling_agent/function_calling_agent_mlflow_sdk

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Evaluate the Agent's quality
# MAGIC
# MAGIC Once you have modified the code & config to create a version of your Agent, there are 3 ways you can test it's quality.
# MAGIC
# MAGIC Each mode does the same high level steps:
# MAGIC 1. Log the Agent's code and config to an MLflow Run --> this captures your code/config should you need to return to this version later after modifying the notebook
# MAGIC 2. Runs the Agent for 1+ queries
# MAGIC 3. Allows you to inspect the Agent's outputs for those queries
# MAGIC
# MAGIC To get started, pick a mode and scroll to the relevant cells below.  If this is your first time, start with 🅰.
# MAGIC
# MAGIC - 🅰 Vibe check the Agent for a single query
# MAGIC   - Use this mode in your inner dev loop to iterate and debug on a single query while making a change.
# MAGIC - 🅱 Evaluate the Agent for 1+ queries
# MAGIC   - Use this mode before you have an evaluation set defined, but want to test a version of the Agent against multiple queries to ensure your change doesn't cause a regression for other queries.
# MAGIC - 🅲 Evaluate the Agent using your evaluation set
# MAGIC   - Use this mode once you have an evaluation set defined.  It is the same as 🅱, but uses your evaluation set that is stored in a Delta Table.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 🚫✏️ Helper function to log the Agent to MLflow
# MAGIC
# MAGIC This helper function wraps the code required to log a version of the Agent's code & config to MLflow.  It is used by all 3 modes.


# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ 🅰 Vibe check the Agent for a single query
# MAGIC
# MAGIC Running this cell will produce an MLflow Trace that you can use to see the Agent's outputs and understand the steps it took to produce that output.

# COMMAND ----------

# Query 
vibe_check_query = {
    "messages": [
        {"role": "user", "content": f"what is lakehouse monitoring?"},
    ]
}

def log_agent_to_mlflow():
    resource_dependencies = get_agent_dependencies(agent_config=agent_config)
    return log_pyfunc_agent(resource_dependencies=resource_dependencies, agent_definition_file_path="agents/function_calling_agent/function_calling_agent_mlflow_sdk", input_example=agent_config.input_example)

# `run_name` provides a human-readable name for this vibe check in the MLflow experiment
with mlflow.start_run(run_name="vibe-check__"+datetime.now().strftime("%Y-%m-%d_%I:%M:%S_%p")):
    # Log the current Agent code/config to MLflow
    logged_agent_info = log_agent_to_mlflow()

    # Execute the Agent
    agent = FunctionCallingAgent(agent_config = agent_config)

    # Run the agent for this query
    response = agent.predict(model_input=vibe_check_query)

    # Print Agent's output
    display_markdown(f"### Agent's output:\n{response['content']}", raw=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ 🅱 Evaluate the Agent for 1+ queries
# MAGIC
# MAGIC Running this cell will call [Agent Evaluation](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) which will run your Agent to generate outputs for 1+ queries and then evaluate the Agent's quality and assess the root cause of any quality issues. The resulting outputs, MLflow Trace, and evaluation results are available in the MLflow Run.

# COMMAND ----------

import pandas as pd

evaluation_set = [
    { # query 1
        "request": {
            "messages": [
                {"role": "user", "content": f"what is lakehouse monitoring?"},
            ]
        }
    },
    { # query 2
        "request": {
            "messages": [
                {"role": "user", "content": f"what is rag?"},
            ]
        }
    }, 
    # add more queries here
]

# `run_name` provides a human-readable name for this vibe check in the MLflow experiment
with mlflow.start_run(run_name="vibe-check__"+datetime.now().strftime("%Y-%m-%d_%I:%M:%S_%p")):
    # Log the current Agent code/config to MLflow
    logged_agent_info = log_agent_to_mlflow()

    # Run the agent for these queries, using Agent evaluation to parallelize the calls
    eval_results = mlflow.evaluate(
        model=logged_agent_info.model_uri, # use the logged Agent
        data=pd.DataFrame(evaluation_set), # Run the logged Agent for all queries defined above
        model_type="databricks-agent", # use Agent Evaluation
    )

    # Show all outputs.  Click on a row in this table to display the MLflow Trace.
    display(eval_results.tables['eval_results'])

    # Click 'View Evaluation Results' to see the Agent's inputs/outputs + quality evaluation displayed in a UI

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ 🅲 Evaluate the Agent using your evaluation set
# MAGIC
# MAGIC Note: If this is your first time creating this agent, this cell will not work.  The evaluation set is populated in the next notebooks using stakeholder feedback.

# COMMAND ----------

# Load the evaluation set from Delta Table
evaluation_set = spark.table(cookbook_shared_config.evaluation_set_table).toPandas()

# `run_name` provides a human-readable name for this vibe check in the MLflow experiment
with mlflow.start_run(run_name="evaluation__"+datetime.now().strftime("%Y-%m-%d_%I:%M:%S_%p")):
    # Log the current Agent code/config to MLflow
    logged_agent_info = log_agent_to_mlflow()

    # Run the agent for these queries, using Agent evaluation to parallelize the calls
    eval_results = mlflow.evaluate(
        model=logged_agent_info.model_uri, # use the logged Agent
        data=evaluation_set, # Run the logged Agent for all queries defined above
        model_type="databricks-agent", # use Agent Evaluation
    )

    # Show all outputs.  Click on a row in this table to display the MLflow Trace.
    display(eval_results.tables['eval_results'])

    # Click 'View Evaluation Results' to see the Agent's inputs/outputs + quality evaluation displayed in a UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Register a version of the Agent to Unity Catalog
# MAGIC
# MAGIC Once you have a version of your Agent that has sufficient quality, you will register the Agent's model from the MLflow Experiment into the Unity Catalog.  This allows you to use the next notebooks to deploy the Agent to Agent Evaluation's Review App to share it with stakeholders & collect feedback.
# MAGIC
# MAGIC You can register a version in two ways:
# MAGIC 1. Register an Agent version that you logged above
# MAGIC 2. Log the latest version of the Agent and then register it

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### ✅✏️ Option 1. Register an Agent that you logged above
# MAGIC
# MAGIC 1. Set the MLflow model's URI in the below cell by either
# MAGIC   - *(Suggested)* If you haven't modified the above code, the `model_uri` from the last logged Agent is stored in the local variable `logged_agent_info.model_uri`.
# MAGIC   - If you want to register a different version:
# MAGIC     - Go the MLflow experiment UI, click on the Run containing the Agent version, and find the Run ID in the Overview tab --> Details section.  
# MAGIC     - Your `model_uri` is `runs:/b5b9436a56544263a97ddd2293e6f422/agent` where `b5b9436a56544263a97ddd2293e6f422` is the Run ID.
# MAGIC
# MAGIC 2. Run the below cell to register the model to Unity Catalog
# MAGIC 3. Note the version of the Unity Catalog model - you will need this in the next notebook to deploy this Agent.

# COMMAND ----------

# Enter the model_uri of the Agent version to be registered
model_uri_to_register = logged_agent_info.model_uri # last Agent logged

# Register a different Agent version
# model_uri_to_register = "runs:/run_id_goes_here/agent" # pick an Agent version

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Register the Agent's model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=model_uri_to_register, name=cookbook_shared_config.uc_model
)

# Print the version number
display_markdown(f"### Unity Catalog model version: **{uc_registered_model_info.version}**", raw=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### ✅✏️ Option 2. Log the latest version of the Agent and register it. 
# MAGIC
# MAGIC 1. Optionally, give the version a short name in `agent_version_short_name` so you can easily identify it in the MLflow experiment later
# MAGIC 1. Run the below cell to log the Agent to a model inside an MLflow Run & register that model to Unity Catalog
# MAGIC 2. Note the version of the Unity Catalog model - you will need this in the next notebook to deploy this Agent.

# COMMAND ----------

agent_version_short_name = "friendly-name-to-identify-this-version" # set to None if you want MLflow to generate a name e.g., `aged-perch-556`

with mlflow.start_run(run_name=agent_version_short_name):
    # Log the current Agent code/config to MLflow
    logged_agent_info = log_agent_to_mlflow()

    # Use Unity Catalog as the model registry
    mlflow.set_registry_uri("databricks-uc")

    # Register this model to the Unity Catalog
    uc_registered_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, name=cookbook_shared_config.uc_model
    )

    # Print the version number
    display_markdown(f"### Unity Catalog model version: **{uc_registered_model_info.version}**", raw=True)
