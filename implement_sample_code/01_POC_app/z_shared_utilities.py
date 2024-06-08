# Databricks notebook source
from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    for key, item in flat_config.items():
        
        sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("{key.replace("/", "__")}" = "{item}")
        """)
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_poc_pdf")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)

# COMMAND ----------

def deduplicate_assessments_table(assessment_table):
    # De-dup response assessments
    assessments_request_deduplicated_df = spark.sql(
        f"""select * except(row_number)
                                        from
                                        (
                                            select
                                            *,
                                            row_number() over (
                                                partition by request_id
                                                order by
                                                timestamp desc
                                            ) as row_number
                                            from
                                            {assessment_table}
                                            where text_assessment is not NULL
                                        )
                                        where
                                        row_number = 1"""
    )
    # De-dup the retrieval assessments
    assessments_retrieval_deduplicated_df = spark.sql(
        f"""select
    *
  except(
      retrieval_assessment,
      source,
      timestamp,
      text_assessment
    ),
    any_value(timestamp) as timestamp,
    any_value(source) as source,
    collect_list(retrieval_assessment) as retrieval_assessments
  from
    {assessment_table}
  where
    retrieval_assessment is not NULL
  group by
    request_id,
    source.id,
    step_id"""
    )

    # Merge together
    assessments_request_deduplicated_df = assessments_request_deduplicated_df.drop("retrieval_assessment", "step_id")
    assessments_retrieval_deduplicated_df = assessments_retrieval_deduplicated_df.withColumnRenamed("request_id", "request_id2").withColumnRenamed("source", "source2").drop("step_id", "timestamp")

    merged_deduplicated_assessments_df = assessments_request_deduplicated_df.join(
        assessments_retrieval_deduplicated_df,
        (assessments_request_deduplicated_df.request_id == assessments_retrieval_deduplicated_df.request_id2) &
        (assessments_request_deduplicated_df.source.id == assessments_retrieval_deduplicated_df.source2.id),
        "full"
    ).select(
        [str(col) for col in assessments_request_deduplicated_df.columns] +
        [assessments_retrieval_deduplicated_df.retrieval_assessments]
    )

    return merged_deduplicated_assessments_df
