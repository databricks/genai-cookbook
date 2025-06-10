from typing import TypedDict
from datetime import datetime
import warnings
import traceback
import os
from urllib.parse import urlparse

# PDF libraries
import fitz
import pymupdf4llm

# HTML libraries
import markdownify
import re

## DOCX libraries
import pypandoc
import tempfile

## JSON libraries
import json


# Schema of the dict returned by `file_parser(...)`
# This is used to create the output Delta Table's schema.
# Adjust the class if you want to add additional columns from your parser, such as extracting custom metadata.
class ParserReturnValue(TypedDict):
    # DO NOT CHANGE THESE NAMES
    # Parsed content of the document
    content: str  # do not change this name
    # The status of whether the parser succeeds or fails, used to exclude failed files downstream
    parser_status: str  # do not change this name
    # Unique ID of the document
    doc_uri: str  # do not change this name

    # OK TO CHANGE THESE NAMES
    # Optionally, you can add additional metadata fields here
    # example_metadata: str
    last_modified: datetime


# Parser function.  Adjust this function to modify the parsing logic.
def file_parser(
    raw_doc_contents_bytes: bytes,
    doc_path: str,
    modification_time: datetime,
    doc_bytes_length: int,
) -> ParserReturnValue:
    """
    Parses the content of a PDF document into a string.

    This function takes the raw bytes of a PDF document and its path, attempts to parse the document using PyPDF,
    and returns the parsed content and the status of the parsing operation.

    Parameters:
    - raw_doc_contents_bytes (bytes): The raw bytes of the document to be parsed (set by Spark when loading the file)
    - doc_path (str): The DBFS path of the document, used to verify the file extension (set by Spark when loading the file)
    - modification_time (timestamp): The last modification time of the document (set by Spark when loading the file)
    - doc_bytes_length (long): The size of the document in bytes (set by Spark when loading the file)

    Returns:
    - ParserReturnValue: A dictionary containing the parsed document content and the status of the parsing operation.
      The 'contenty will contain the parsed text as a string, and the 'parser_status' key will indicate
      whether the parsing was successful or if an error occurred.
    """
    try:
        from markdownify import markdownify as md

        filename, file_extension = os.path.splitext(doc_path)

        if file_extension == ".pdf":
            pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(pdf_doc)

            parsed_document = {
                "content": md_text.strip(),
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".html":
            html_content = raw_doc_contents_bytes.decode("utf-8")

            markdown_contents = md(
                str(html_content).strip(), heading_style=markdownify.ATX
            )
            markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

            parsed_document = {
                "content": markdown_stripped,
                "parser_status": "SUCCESS",
            }
        elif file_extension == ".docx":
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_file.write(raw_doc_contents_bytes)
                temp_file_path = temp_file.name
                md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

                parsed_document = {
                    "content": md.strip(),
                    "parser_status": "SUCCESS",
                }
        elif file_extension in [".txt", ".md"]:
            parsed_document = {
                "content": raw_doc_contents_bytes.decode("utf-8").strip(),
                "parser_status": "SUCCESS",
            }
        elif file_extension in [".json", ".jsonl"]:
            # NOTE: This is a placeholder for a JSON parser.  It's not a "real" parser, it just returns the raw JSON formatted into XML-like strings that LLMs tend to like.
            json_data = json.loads(raw_doc_contents_bytes.decode("utf-8"))

            def flatten_json_to_xml(obj, parent_key=""):
                xml_parts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            xml_parts.append(flatten_json_to_xml(value, key))
                        else:
                            xml_parts.append(f"<{key}>{str(value)}</{key}>")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            xml_parts.append(
                                flatten_json_to_xml(item, f"{parent_key}_{i}")
                            )
                        else:
                            xml_parts.append(
                                f"<{parent_key}_{i}>{str(item)}</{parent_key}_{i}>"
                            )
                else:
                    xml_parts.append(f"<{parent_key}>{str(obj)}</{parent_key}>")
                return "\n".join(xml_parts)

            flattened_content = flatten_json_to_xml(json_data)
            parsed_document = {
                "content": flattened_content.strip(),
                "parser_status": "SUCCESS",
            }
        else:
            raise Exception(f"No supported parser for {doc_path}")

        # Extract the required doc_uri
        # convert from `dbfs:/Volumes/catalog/schema/pdf_docs/filename.pdf` to `/Volumes/catalog/schema/pdf_docs/filename.pdf`
        modified_path = urlparse(doc_path).path
        parsed_document["doc_uri"] = modified_path

        # Sample metadata extraction logic
        # if "test" in parsed_document["content
        #     parsed_document["example_metadata"] = "test"
        # else:
        #     parsed_document["example_metadata"] = "not test"

        # Add the modified time
        parsed_document["last_modified"] = modification_time

        return parsed_document

    except Exception as e:
        status = f"An error occurred: {e}\n{traceback.format_exc()}"
        warnings.warn(status)
        return {
            "content": "",
            "parser_status": f"ERROR: {status}",
        }
