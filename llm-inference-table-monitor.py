# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Inference Table Analysis With Text Evaluation Metrics Computation and Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-0.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC #### About this notebook
# MAGIC This starter notebook is intended to be used with **Databricks Model Serving** endpoints which have the *Inference Table* feature enabled. To set up a generation endpoint, refer to the guide on model serving endpoints ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/score-model-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-model-serving-endpoints)).</br>
# MAGIC This notebook has three high-level purposes:
# MAGIC
# MAGIC 1. Unpack the logged requests and responses by converting your model raw JSON payloads as string.
# MAGIC 2. Compute text evaluation metrics over the extracted input/output.
# MAGIC 3. Setup Databricks Lakehouse Monitoring on the resulting table to produce data and model quality/drift metrics.
# MAGIC
# MAGIC #### How to run the notebook
# MAGIC The notebook is set up to be run step-by-step. Here are the main configuration to set:
# MAGIC * Define your model serving endpoint name (mandatory)
# MAGIC * Ensure the unpacking function works with your model input/output schema
# MAGIC * Define the checkpoint location (prefer using a Volume within your schema)
# MAGIC For best results, run this notebook on any cluster running **Machine Learning Runtime 12.2LTS or higher**.
# MAGIC
# MAGIC #### Scheduling
# MAGIC Feel free to run this notebook manually to test out the parameters; when you're ready to run it in production, you can schedule it as a recurring job.</br>
# MAGIC Note that in order to keep this notebook running smoothly and efficiently, we recommend running it at least **once a week** to keep output tables fresh and up to date.

# COMMAND ----------

# DBTITLE 1,Load the required libraries
# MAGIC %pip install textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.30.2 torch==1.13.1 "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.14-py3-none-any.whl" jinja2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exploring the Model Serving Inference table content
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-inference-table.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC Let's start by analyzing what's inside our inference table.
# MAGIC
# MAGIC The inference table name can be fetched from the model serving endpoint configuration. 
# MAGIC
# MAGIC We'll first get the table name and simply run a query to view its content.

# COMMAND ----------

# Set widgets for required parameters for this notebook.
dbutils.widgets.text("endpoint", "", label = "Name of Model Serving Endpoint")
endpoint_name = dbutils.widgets.get("endpoint")
if len(endpoint_name) == 0:
    raise Exception("Please fill in the required information for endpoint name.")


# Location to store streaming checkpoint
dbutils.widgets.text("checkpoint_location", f"dbfs:/payload-logging/{endpoint_name}/checkpoint", label = "Checkpoint Location")
checkpoint_location = dbutils.widgets.get("checkpoint_location")

# COMMAND ----------

import requests
from typing import Dict
from pyspark.sql.functions import element_at

def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name).limit(10)
display(payloads)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Unpacking the inference table requests and responses and computing the LLM metrics
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-1.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC ### Unpacking the table
# MAGIC
# MAGIC The request and response columns contains your model input and output as a `string`.
# MAGIC
# MAGIC Note that the format depends of your model definition and can be custom. Inputs are usually represented as JSON with TF format, and the output depends of your model definition.
# MAGIC
# MAGIC Because our model is designed to potentially batch multiple entries, we need to unpack the value from the request and response.
# MAGIC
# MAGIC We will use Spark JSON Path annotation to directly access the query and response as string, concatenate the input/output together with an `array_zip` and ultimately `explode` the content to have 1 input/output per line (unpacking the batches)
# MAGIC
# MAGIC **Make sure you change the following selectors based on your model definition**
# MAGIC
# MAGIC *Note: This will be made easier within the product directly, we provide this notebook to simplify this task for now.*

# COMMAND ----------

# DBTITLE 1,Define the Json Path to extract the input and output values
# The format of the input payloads.
INPUT_REQUEST_JSON_PATH = "messages[*].content"
# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
INPUT_JSON_PATH_TYPE = "array<string>"

# Answer format:
OUTPUT_REQUEST_JSON_PATH = "choices[*].message.content"
# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import pandas_udf


def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str) -> DataFrame:
    # Convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumn("__db_timestamp", (F.col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "__db_model_id",
        F.concat(
            F.col("request_metadata").getItem("model_name"),
            F.lit("_"),
            F.col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(F.col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", element_at(F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type), -1))
        .withColumn("response", element_at(F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type), -1)))

    # Explode batched requests into individual rows.
    # requests_exploded = (requests_unpacked
    #     .withColumn("__db_request_response", F.explode(F.arrays_zip(F.col("request").alias("input"), F.col("response").alias("output"))))
    #     .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
    #     )

    return requests_unpacked

# Let's try our unpacking function. Make sure input & output columns are not null
display(unpack_requests(payloads, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute the Input / Output text evaluation metrics (e.g., toxicity, perplexity, readability) 
# MAGIC
# MAGIC Now that our input and output are unpacked and available as a string, we can compute their metrics. These will be analyzed by Lakehouse Monitoring so that we can understand how these metrics change over time.
# MAGIC
# MAGIC Feel free to add your own custom evaluation metrics here.

# COMMAND ----------

import tiktoken, textstat, evaluate
import pandas as pd


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

# COMMAND ----------

def compute_metrics(requests_df: DataFrame, column_to_measure = ["request", "response"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"toxicity({column_name})", compute_toxicity(F.col(column_name)))
                 .withColumn(f"perplexity({column_name})", compute_perplexity(F.col(column_name)))
                 .withColumn(f"token_count({column_name})", compute_num_tokens(F.col(column_name)))
                 .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(F.col(column_name)))
                 .withColumn(f"automated_readability_index({column_name})", automated_readability_index(F.col(column_name)))
    )
  return requests_df

# Initialize the processed requests table. Turn on CDF (for monitoring) and enable special characters in column names. 
def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    (DeltaTable.createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name") \
        .property("delta.minReaderVersion", "2") \
        .property("delta.minWriterVersion", "5")
        .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC We can now incrementally consume new payload from the inference table, unpack them, compute metrics and save them to our final processed table:

# COMMAND ----------

from delta.tables import DeltaTable


# Check whether the table exists before proceeding.
DeltaTable.forName(spark, payload_table_name)

# Unpack the requests as a stream.
requests_raw = spark.readStream.table(payload_table_name)
requests_processed = unpack_requests(requests_raw, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE).filter(F.col("request").isNotNull())

# Drop columns that we don't need for monitoring analysis.
requests_processed = requests_processed.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

# Compute text evaluation metrics.
requests_with_metrics = compute_metrics(requests_processed)

# Persist the requests stream, with a defined checkpoint path for this table.
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics)
(requests_with_metrics.writeStream
                      .trigger(availableNow=True)
                      .format("delta")
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name).awaitTermination())

# Display the table (with requests and text evaluation metrics) that will be monitored.
display(spark.table(processed_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Monitor the inference table
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-2.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC In this step, we create a monitor on our inference table by using the `create_monitor` API. If the monitor already exists, we pass the same parameters to `update_monitor`. In steady state, this should result in no change to the monitor.
# MAGIC
# MAGIC Afterwards, we queue a metric refresh so that the monitor analyzes the latest processed requests.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html) | [Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.

# COMMAND ----------

"""
Optional parameters to control monitoring analysis. For help, use the command help(lm.create_monitor).
"""
GRANULARITIES = ["1 day"]                        # Window sizes to analyze data over
SLICING_EXPRS = None                             # Expressions to slice data with

CUSTOM_METRICS = None                            # A list of custom metrics to compute
BASELINE_TABLE = None                            # Baseline table name, if any, for computing baseline drift

# COMMAND ----------

import databricks.lakehouse_monitoring as lm


monitor_params = {
    "profile_type": lm.TimeSeries(
        timestamp_col="__db_timestamp",
        granularities=GRANULARITIES,
    ),
    "output_schema_name": f"{catalog}.{schema}",
    "schedule": None,  # We will refresh the metrics on-demand in this notebook
    "baseline_table_name": BASELINE_TABLE,
    "slicing_exprs": SLICING_EXPRS,
    "custom_metrics": CUSTOM_METRICS,
}

try:
    info = lm.create_monitor(table_name=processed_table_name, **monitor_params)
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(table_name=processed_table_name, updated_params=monitor_params)
    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_table_name)
    print(refresh_info)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our table is now monitored
# MAGIC
# MAGIC Databricks Lakehouse Monitoring automatically builds dashboard to track your metrics and their evolution over time.
# MAGIC
# MAGIC You can leverage your metric table to track your LLM model behavior over time, and setup alerts to detect potential changes in model perplexity or toxicity.
