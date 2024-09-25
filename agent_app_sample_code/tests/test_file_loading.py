import pytest

# TODO: dedup with data pipeline
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES - these are required by Evaluation & Framework
    # Parsed content of the document
    doc_content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name

    # OK TO CHANGE THESE NAMES
    # Optionally, you can add additional metadata fields here
    example_metadata: str
    last_modified: datetime

def test_load_uc_volume_to_delta_table():
    assert True
    def dummy_file_parser(
            raw_doc_contents_bytes: bytes,
            doc_path: str,
            modification_time: datetime,
            doc_bytes_length: int,
    ) -> ParserReturnValue:

    #
    load_uc_volume_to_delta_table(
        spark=spark,
        source_path=SOURCE_UC_VOLUME,
        dest_table_name=DOCS_DELTA_TABLE,
        # Modify this function to change the parser, extract additional metadata, etc
        parse_file_udf=file_parser,
        # The schema of the resulting Delta Table will follow the schema defined in ParserReturnValue
        spark_dataframe_schema=typed_dicts_to_spark_schema(ParserReturnValue),
    )
