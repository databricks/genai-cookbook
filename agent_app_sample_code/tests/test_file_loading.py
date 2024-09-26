import pytest
import pyspark
import pandas as pd

from agent_app_sample_code.utils.file_loading import load_files_to_df

@pytest.fixture(scope="module")
def spark():
    return (
        pyspark.sql.SparkSession.builder
        .master("local[1]")
        # Uncomment the following line for testing on Apple silicon locally
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.task.maxFailures", "1")  # avoid retry failed spark tasks
        .getOrCreate()
            )

def test_load_files_to_df(spark, tmpdir):
    temp_dir = tmpdir.mkdir("files_subdir")
    file_1 = temp_dir.join("file1.txt")
    file_2 = temp_dir.join("file2.txt")
    file_1.write("file1 content")
    file_2.write("file2 content")
    raw_files_df = load_files_to_df(spark, str(temp_dir)).drop("modificationTime").orderBy("path")
    assert raw_files_df.count() == 2
    raw_pandas_df = raw_files_df.toPandas()
    # Decode the content from bytes to string
    raw_pandas_df['content'] = raw_pandas_df['content'].apply(
        lambda x: bytes(x).decode('utf-8')
    )
    # Expected DataFrame
    expected_df = pd.DataFrame([{
        "path": f"file:{str(file_1)}",
        "length": len("file1 content"),
        "content": "file1 content",
    }, {
        "path": f"file:{str(file_2)}",
        "length": len("file2 content"),
        "content": "file2 content",
    }])
    pd.testing.assert_frame_equal(raw_pandas_df, expected_df)

def test_load_files_to_df_throws_if_no_files(spark, tmpdir):
    temp_dir = tmpdir.mkdir("files_subdir")
    with pytest.raises(Exception, match="does not contain any files"):
        load_files_to_df(spark, str(temp_dir))
