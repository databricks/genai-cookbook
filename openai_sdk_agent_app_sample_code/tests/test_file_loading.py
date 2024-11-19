from datetime import datetime

import pytest
import pyspark
import pandas as pd
from typing import TypedDict

from utils.file_loading import load_files_to_df, apply_parsing_udf
from utils.typed_dicts_to_spark_schema import typed_dicts_to_spark_schema


@pytest.fixture(scope="module")
def spark():
    return (
        pyspark.sql.SparkSession.builder.master("local[1]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.task.maxFailures", "1")  # avoid retry failed spark tasks
        .getOrCreate()
    )


@pytest.fixture()
def example_files_dir(tmpdir):
    temp_dir = tmpdir.mkdir("files_subdir")
    file_1 = temp_dir.join("file1.txt")
    file_2 = temp_dir.join("file2.txt")
    file_1.write("file1 content")
    file_2.write("file2 content")
    yield temp_dir, file_1, file_2


def test_load_files_to_df(spark, example_files_dir):
    temp_dir, file_1, file_2 = example_files_dir
    raw_files_df = (
        load_files_to_df(spark, str(temp_dir)).drop("modificationTime").orderBy("path")
    )
    assert raw_files_df.count() == 2
    raw_pandas_df = raw_files_df.toPandas()
    # Decode the content from bytes to string
    raw_pandas_df["content"] = raw_pandas_df["content"].apply(
        lambda x: bytes(x).decode("utf-8")
    )
    # Expected DataFrame
    expected_df = pd.DataFrame(
        [
            {
                "path": f"file:{str(file_1)}",
                "length": len("file1 content"),
                "content": "file1 content",
            },
            {
                "path": f"file:{str(file_2)}",
                "length": len("file2 content"),
                "content": "file2 content",
            },
        ]
    )
    pd.testing.assert_frame_equal(raw_pandas_df, expected_df)


def test_load_files_to_df_throws_if_no_files(spark, tmpdir):
    temp_dir = tmpdir.mkdir("files_subdir")
    with pytest.raises(Exception, match="does not contain any files"):
        load_files_to_df(spark, str(temp_dir))


class ParserReturnValue(TypedDict):
    # Parsed content of the document
    content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name


def test_apply_parsing_udf(spark, example_files_dir):
    def _mock_file_parser(
        raw_doc_contents_bytes: bytes,
        doc_path: str,
        modification_time: datetime,
        doc_bytes_length: int,
    ):
        return {
            "content": raw_doc_contents_bytes.decode("utf-8"),
            "parser_status": "SUCCESS",
            "doc_uri": doc_path,
        }

    temp_dir, file_1, file_2 = example_files_dir
    raw_files_df = load_files_to_df(spark, str(temp_dir)).orderBy("path")
    parsed_df = apply_parsing_udf(
        raw_files_df,
        _mock_file_parser,
        parsed_df_schema=typed_dicts_to_spark_schema(ParserReturnValue),
    )
    assert parsed_df.count() == 2
    parsed_pandas_df = parsed_df.toPandas()
    # Expected DataFrame
    expected_df = pd.DataFrame(
        [
            {
                "content": file_1.read_text(encoding="utf-8"),
                "parser_status": "SUCCESS",
                "doc_uri": f"file:{str(file_1)}",
            },
            {
                "content": file_2.read_text(encoding="utf-8"),
                "parser_status": "SUCCESS",
                "doc_uri": f"file:{str(file_2)}",
            },
        ]
    )
    pd.testing.assert_frame_equal(parsed_pandas_df, expected_df)
