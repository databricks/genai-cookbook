# Databricks notebook source
import pandas as pd
from typing import List, Mapping, Optional

import mlflow
import mlflow.entities as mlflow_entities

from pyspark import sql
from pyspark.sql import functions as F, types as T
from pyspark.sql.window import Window

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.rag_eval.evaluation import traces


# COMMAND ----------

# Deduplicate the assessment log

# By default, the assessment log contains one row for every action/click the user does in the Review App.  This code translates these logs into a single row for each request.

_REQUEST_ID = "request_id"
_TIMESTAMP = "timestamp"
_ROW_NUMBER = "row_number"
_SOURCE = "source"
_SOURCE_ID = "source.id"
_STEP_ID = "step_id"
_TEXT_ASSESSMENT = "text_assessment"
_RETRIEVAL_ASSESSMENT = "retrieval_assessment"


def _dedup_by_assessment_window(
    assessment_log_df: sql.DataFrame, window: Window
) -> sql.DataFrame:
    """
    Dedup the assessment logs by taking the first row from each group, defined by the window
    :param assessment_log_df: Pyspark DataFrame of the assessment logs
    :param window: Pyspark window to group assessments by
    :return: Pyspark DataFrame of the deduped assessment logs
    """
    return (
        assessment_log_df.withColumn(_ROW_NUMBER, F.row_number().over(window))
        .filter(F.col(_ROW_NUMBER) == 1)
        .drop(_ROW_NUMBER)
    )


def _dedup_assessment_log(assessment_log_df: sql.DataFrame) -> sql.DataFrame:
    """
    Dedup the assessment logs to get the latest assessments.
    :param assessment_log_df: Pyspark DataFrame of the assessment logs
    :return: Pyspark DataFrame of the deduped assessment logs
    """
    # Dedup the text assessments
    text_assessment_window = Window.partitionBy(_REQUEST_ID, _SOURCE_ID).orderBy(
        F.col(_TIMESTAMP).desc()
    )
    deduped_text_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null text assessments
        assessment_log_df.filter(F.col(_TEXT_ASSESSMENT).isNotNull()),
        text_assessment_window,
    )

    # Dedup the retrieval assessments
    retrieval_assessment_window = Window.partitionBy(
        _REQUEST_ID,
        _SOURCE_ID,
        f"{_RETRIEVAL_ASSESSMENT}.position",
        f"{_RETRIEVAL_ASSESSMENT}.{_STEP_ID}",
    ).orderBy(F.col(_TIMESTAMP).desc())
    deduped_retrieval_assessment_df = _dedup_by_assessment_window(
        # Filter rows with null retrieval assessments
        assessment_log_df.filter(F.col(_RETRIEVAL_ASSESSMENT).isNotNull()),
        retrieval_assessment_window,
    )

    # Collect retrieval assessments from the same request/step/source into a single list
    nested_retrieval_assessment_df = (
        deduped_retrieval_assessment_df.groupBy(_REQUEST_ID, _SOURCE_ID, _STEP_ID).agg(
            F.any_value(_TIMESTAMP).alias(_TIMESTAMP),
            F.any_value(_SOURCE).alias(_SOURCE),
            F.collect_list(_RETRIEVAL_ASSESSMENT).alias("retrieval_assessments"),
        )
        # Drop the old retrieval assessment, source id, and text assessment columns
        .drop(_RETRIEVAL_ASSESSMENT, "id", _TEXT_ASSESSMENT)
    )

    # Join the deduped text assessments with the nested deduped retrieval assessments
    deduped_assessment_log_df = deduped_text_assessment_df.alias("a").join(
        nested_retrieval_assessment_df.alias("b"),
        (F.col(f"a.{_REQUEST_ID}") == F.col(f"b.{_REQUEST_ID}"))
        & (F.col(f"a.{_SOURCE_ID}") == F.col(f"b.{_SOURCE_ID}")),
        "full_outer",
    )

    # Coalesce columns from both dataframes in case a request does not have either assessment
    return deduped_assessment_log_df.select(
        F.coalesce(F.col(f"a.{_REQUEST_ID}"), F.col(f"b.{_REQUEST_ID}")).alias(
            _REQUEST_ID
        ),
        F.coalesce(F.col(f"a.{_STEP_ID}"), F.col(f"b.{_STEP_ID}")).alias(_STEP_ID),
        F.coalesce(F.col(f"a.{_TIMESTAMP}"), F.col(f"b.{_TIMESTAMP}")).alias(
            _TIMESTAMP
        ),
        F.coalesce(F.col(f"a.{_SOURCE}"), F.col(f"b.{_SOURCE}")).alias(_SOURCE),
        F.col(f"a.{_TEXT_ASSESSMENT}").alias(_TEXT_ASSESSMENT),
        F.col("b.retrieval_assessments").alias(_RETRIEVAL_ASSESSMENT),
        # F.col("schema_version")
    )

# COMMAND ----------

## Attach ground truth


def attach_ground_truth(request_log_df, deduped_assessment_log_df):
    suggested_output_col = F.col(f"{_TEXT_ASSESSMENT}.suggested_output")
    is_correct_col = F.col(f"{_TEXT_ASSESSMENT}.ratings.answer_correct.value")
    # Extract out the thumbs up/down rating and the suggested output
    rating_log_df = (
        deduped_assessment_log_df.withColumn("is_correct", is_correct_col)
        .withColumn(
            "suggested_output",
            F.when(suggested_output_col == "", None).otherwise(suggested_output_col),
        ).withColumn("source_user", F.col("source.id"))
        .select("request_id", "is_correct", "suggested_output", "source_user", _RETRIEVAL_ASSESSMENT)
    )
    # Join the request log with the ratings from above
    raw_requests_with_feedback_df = request_log_df.join(
        rating_log_df,
        request_log_df.databricks_request_id == rating_log_df.request_id,
        "left",
    )

    raw_requests_with_feedback_df = raw_requests_with_feedback_df.drop("request_id")
    return raw_requests_with_feedback_df



# COMMAND ----------

_EXPECTED_RETRIEVAL_CONTEXT_SCHEMA = T.ArrayType(T.StructType([T.StructField("doc_uri", T.StringType()), T.StructField("content", T.StringType())]))

def extract_retrieved_chunks_from_trace(trace_str: str) -> List[Mapping[str, str]]:
  """Helper to extract the retrieved chunks from a trace string"""
  trace = mlflow_entities.Trace.from_json(trace_str)
  chunks = traces.extract_retrieval_context_from_trace(trace)
  return [{"doc_uri": chunk.doc_uri, "content": chunk.content} for chunk in chunks]

@F.udf(_EXPECTED_RETRIEVAL_CONTEXT_SCHEMA)
def construct_expected_retrieval_context(trace_str: Optional[str], chunk_at_i_relevance: Optional[List[str]]) -> Optional[List[Mapping[str, str]]]:
  """Helper to construct the expected retrieval context. Any retrieved chunks that are not relevant are dropped."""
  if chunk_at_i_relevance is None or trace_str is None: 
    return None
  retrieved_chunks = extract_retrieved_chunks_from_trace(trace_str)
  expected_retrieval_context = [chunk for chunk, rating in zip(retrieved_chunks, chunk_at_i_relevance) if rating == "true"]
  return expected_retrieval_context if len(expected_retrieval_context) else None
# =================================


def identify_potential_eval_set_records(raw_requests_with_feedback_df):
  # For thumbs up, use either the suggested output or the response, in that order
  positive_feedback_df = (
    raw_requests_with_feedback_df
      .where(F.col("is_correct") == F.lit("positive"))
      .withColumn(
        "expected_response",
        F.when(
          F.col("suggested_output") != None, F.col("suggested_output")
        ).otherwise(F.col("response"))
      )
      .withColumn("source_tag", F.lit("thumbs_up"))
  )

  # For thumbs down, use the suggested output if there is one
  negative_feedback_df = (
    raw_requests_with_feedback_df
      .where(F.col("is_correct") == F.lit("negative"))
      .withColumn("expected_response", F.col("suggested_output"))
      .withColumn("source_tag", F.lit("thumbs_down_edited"))
  )

  # For no feedback or IDK, there is no expected response.
  no_or_unknown_feedback_df = (
    raw_requests_with_feedback_df
      .where((F.col("is_correct").isNull()) | ((F.col("is_correct") != F.lit("negative")) & (F.col("is_correct") != F.lit("positive"))))
      .withColumn("expected_response", F.lit(None))
      .withColumn("source_tag", F.lit("no_feedback_provided"))
  )
  # Join the above feedback tables and select the relevant columns for the eval harness
  requests_with_feedback_df = positive_feedback_df.unionByName(negative_feedback_df).unionByName(no_or_unknown_feedback_df)
  # Get the thumbs up/down for each retrieved chunk
  requests_with_feedback_df = requests_with_feedback_df.withColumn(
      "chunk_at_i_relevance",
      F.transform(
          F.col(_RETRIEVAL_ASSESSMENT),
          lambda x: x.ratings.answer_correct.value
      )
  ).drop(_RETRIEVAL_ASSESSMENT)

  requests_with_feedback_df = requests_with_feedback_df.withColumnRenamed("databricks_request_id", "request_id")
  
  # Add the expected retrieved context column
  requests_with_feedback_df = requests_with_feedback_df.withColumn(
    "expected_retrieved_context", construct_expected_retrieval_context(F.col("trace"), F.col("chunk_at_i_relevance"))
  )
  return requests_with_feedback_df
  

# COMMAND ----------

def create_potential_evaluation_set(request_log_df, assessment_log_df):
    raw_requests_with_feedback_df = attach_ground_truth(request_log_df, assessment_log_df)
    requests_with_feedback_df = identify_potential_eval_set_records(raw_requests_with_feedback_df)
    return requests_with_feedback_df
