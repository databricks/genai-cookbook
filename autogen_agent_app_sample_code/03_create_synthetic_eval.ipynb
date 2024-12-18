# Databricks notebook source
# MAGIC %md
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC ### Step 1: Create synthetic evaluation data
# MAGIC
# MAGIC To measure your Agent's quality, you need a diverse, representative evaluation set.  This notebook turns your unstructured documents into a high-quality synthetic evaluation set so that you can start to evaluate and improve your Agent's quality before subject matter experts are available to label data.
# MAGIC
# MAGIC This notebook does the following:
# MAGIC 1. <TODO>
# MAGIC
# MAGIC THIS DOES NOT WORK FROM LOCAL IDE YET.

# COMMAND ----------

# MAGIC %md
# MAGIC **Important note:** Throughout this notebook, we indicate which cells you:
# MAGIC - ✅✏️ *should* customize - these cells contain config settings to change
# MAGIC - 🚫✏️ *typically will not* customize - these cells contain  code that is parameterized by your configuration.
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Install Python libraries

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Connect to Databricks
# MAGIC
# MAGIC If running locally in an IDE using Databricks Connect, connect the Spark client & configure MLflow to use Databricks Managed MLflow.  If this running in a Databricks Notebook, these values are already set.

# COMMAND ----------

from mlflow.utils import databricks_utils as du
import os

if not du.is_in_databricks_notebook():
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Load the Agent's storage locations
# MAGIC
# MAGIC This notebook writes to the evaluation set table that you specified in the [Agent setup](02_agent_setup.ipynb) notebook.

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.databricks_utils import get_table_url
from cookbook.config import load_serializable_config_from_yaml_file

# Load the Agent's storage configuration
agent_storage_config: AgentStorageConfig = load_serializable_config_from_yaml_file('./configs/agent_storage_config.yaml')

# Check if the evaluation set already exists
try:
    eval_dataset = spark.table(agent_storage_config.evaluation_set_uc_table)
    if eval_dataset.count() > 0:
        print(f"Evaluation set {get_table_url(agent_storage_config.evaluation_set_uc_table)} already exists!  By default, this notebook will append to the evaluation dataset.  If you would like to overwrite the existing evaluation set, please delete the table before running this notebook.")
    else:
        print(f"Evaluation set {get_table_url(agent_storage_config.evaluation_set_uc_table)} exists, but is empty!  By default, this notebook will NOT change the schema of this table - if you experience schema related errors, drop this table before running this notebook so it can be recreated with the correct schema.")
except Exception:
    print(f"Evaluation set `{agent_storage_config.evaluation_set_uc_table}` does not exist.  This notebook will create a new Delta Table at this location.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Load the source documents for synthetic evaluation data generation
# MAGIC
# MAGIC Most often, this will be the same as the document output table from the [data pipeline](01_data_pipeline.ipynb).
# MAGIC
# MAGIC Here, we provide code to load the documents table that was created in the [data pipeline](01_data_pipeline.ipynb).
# MAGIC
# MAGIC Alternatively, this can be a Spark DataFrame, Pandas DataFrame, or list of dictionaries with the following keys/columns:
# MAGIC - `doc_uri`: A URI pointing to the document.
# MAGIC - `content`: The content of the document.

# COMMAND ----------

from cookbook.config.data_pipeline import DataPipelineConfig
from cookbook.config import load_serializable_config_from_yaml_file

datapipeline_config: DataPipelineConfig= load_serializable_config_from_yaml_file('./configs/data_pipeline_config.yaml')

source_documents = spark.table(datapipeline_config.output.parsed_docs_table)

display(source_documents.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Run the synthetic evaluation data generation
# MAGIC
# MAGIC Optionally, you can customize the guidelines to guide the synthetic data generation.  By default, guidelines are not applied - to apply the guidelines, uncomment `guidelines=guidelines` in the `generate_evals_df(...)` call.  See our [documentation](https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html) for more details.

# COMMAND ----------

from databricks.agents.evals import generate_evals_df

# NOTE: The guidelines you provide are a free-form string. The markdown string below is the suggested formatting for the set of guidelines, however you are free
# to add your sections here. Note that this will be prompt-engineering an LLM that generates the synthetic data, so you may have to iterate on these guidelines before
# you get the results you desire.
guidelines = """
# Task Description
The Agent is a RAG chatbot that answers questions about using Spark on Databricks. The Agent has access to a corpus of Databricks documents, and its task is to answer the user's questions by retrieving the relevant docs from the corpus and synthesizing a helpful, accurate response. The corpus covers a lot of info, but the Agent is specifically designed to interact with Databricks users who have questions about Spark. So questions outside of this scope are considered irrelevant.

# User personas
- A developer who is new to the Databricks platform
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- what API lets me parallelize operations over rows of a delta table?
- Which cluster settings will give me the best performance when using Spark?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

synthesized_evals_df = generate_evals_df(
    docs=source_documents,
    # The number of evaluations to generate for each doc.
    num_evals=10,
    # A optional set of guidelines that help guide the synthetic generation. This is a free-form string that will be used to prompt the generation.
    # guidelines=guidelines
)

# Write the synthetic evaluation data to the evaluation set table
spark.createDataFrame(synthesized_evals_df).write.format("delta").mode("append").saveAsTable(agent_storage_config.evaluation_set_uc_table)

# Display the synthetic evaluation data
eval_set_df = spark.table(agent_storage_config.evaluation_set_uc_table)
display(eval_set_df.toPandas())