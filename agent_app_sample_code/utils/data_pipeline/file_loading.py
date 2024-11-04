import traceback
from datetime import datetime
from typing import Any, Callable, TypedDict, Dict
import os
from IPython.display import display_markdown
import warnings
import pyspark.sql.functions as func
from pyspark.sql.types import StructType
from pyspark.sql import DataFrame, SparkSession


def _parse_and_extract(
    raw_doc_contents_bytes: bytes,
    modification_time: datetime,
    doc_bytes_length: int,
    doc_path: str,
    parse_file_udf: Callable[[[dict, Any]], str],
) -> Dict[str, Any]:
    """Parses raw bytes & extract metadata."""
    try:
        # Run the parser
        parser_output_dict = parse_file_udf(
            raw_doc_contents_bytes=raw_doc_contents_bytes,
            doc_path=doc_path,
            modification_time=modification_time,
            doc_bytes_length=doc_bytes_length,
        )

        if parser_output_dict.get("parser_status") == "SUCCESS":
            return parser_output_dict
        else:
            raise Exception(parser_output_dict.get("parser_status"))

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "doc_content": "",
            "doc_uri": doc_path,
            "parser_status": status,
        }


def _get_parser_udf(
    # extract_metadata_udf: Callable[[[dict, Any]], str],
    parse_file_udf: Callable[[[dict, Any]], str],
    spark_dataframe_schema: StructType,
):
    """Gets the Spark UDF which will parse the files in parallel.

    Arguments:
      - extract_metadata_udf: A function that takes parsed content and extracts the metadata
      - parse_file_udf: A function that takes the raw file and returns the parsed text.
      - spark_dataframe_schema: The resulting schema of the document delta table
    """
    # This UDF will load each file, parse the doc, and extract metadata.
    parser_udf = func.udf(
        lambda raw_doc_contents_bytes, modification_time, doc_bytes_length, doc_path: _parse_and_extract(
            raw_doc_contents_bytes,
            modification_time,
            doc_bytes_length,
            doc_path,
            parse_file_udf,
        ),
        returnType=spark_dataframe_schema,
    )
    return parser_udf


def load_files_to_df(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Load files from a directory into a Spark DataFrame.
    Each row in the DataFrame will contain the path, length, and content of the file; for more
    details, see https://spark.apache.org/docs/latest/sql-data-sources-binaryFile.html
    """

    print(f"Loading the raw files from {source_path}...")
    # Load the raw riles
    raw_files_df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(source_path)
    )

    # Check that files were present and loaded
    if raw_files_df.count() == 0:
        raise Exception(f"`{source_path}` does not contain any files.")

    # display_markdown(
    #     f"### Found {raw_files_df.count()} files in {source_path}: ", raw=True
    # )
    # raw_files_df.display()
    return raw_files_df


def apply_parsing_fn(
    raw_files_df: DataFrame,
    parse_file_fn: Callable[[[dict, Any]], str],
    parsed_df_schema: StructType,
) -> DataFrame:
    """
    Apply a file-parsing UDF to a DataFrame whose rows correspond to file content/metadata loaded via
    https://spark.apache.org/docs/latest/sql-data-sources-binaryFile.html
    Returns a DataFrame with the parsed content and metadata.
    """
    print("Running parsing & metadata extraction UDF in spark...")

    parser_udf = _get_parser_udf(parse_file_fn, parsed_df_schema)

    # Run the parsing
    parsed_files_staging_df = raw_files_df.withColumn(
        "parsing", parser_udf("content", "modificationTime", "length", "path")
    ).drop("content")

    # Filter for successfully parsed files
    parsed_files_df = parsed_files_staging_df  # .filter(
    #    parsed_files_staging_df.parsing.parser_status == "SUCCESS"
    # )

    # Change the schema to the resulting schema
    resulting_fields = [field.name for field in parsed_df_schema.fields]

    parsed_files_df = parsed_files_df.select(
        *[func.col(f"parsing.{field}").alias(field) for field in resulting_fields]
    )
    return parsed_files_df


def check_parsed_df_for_errors(parsed_files_df) -> tuple[bool, str, DataFrame]:
    # Check and warn on any errors
    errors_df = parsed_files_df.filter(func.col(f"parser_status") != "SUCCESS")

    num_errors = errors_df.count()
    if num_errors > 0:
        msg = f"{num_errors} documents ({round(errors_df.count()/parsed_files_df.count(), 2)*100}) of documents had parse errors. Please review."
        return (True, msg, errors_df)
    else:
        msg = "All documents were parsed."
        print(msg)
        return (False, msg, None)


def check_parsed_df_for_empty_parsed_files(parsed_files_df):
    # Check and warn on any errors
    num_empty_df = parsed_files_df.filter(
        func.col(f"parser_status") == "SUCCESS"
    ).filter(func.col("doc_content") == "")

    num_errors = num_empty_df.count()
    if num_errors > 0:
        msg = f"{num_errors} documents ({round(num_empty_df.count()/parsed_files_df.count(), 2)*100}) of documents returned empty parsing results. Please review."
        return (True, msg, num_empty_df)
    else:
        msg = "All documents produced non-null parsing results."
        print(msg)
        return (False, msg, None)
