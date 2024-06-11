# Databricks notebook source
# MAGIC %md
# MAGIC ## Parser Library
# MAGIC
# MAGIC This notebook implements a library of parsing tools for PDF documents. It should be `%run` as the first cell inside the `02_parse_docs` notebook. The `parser_factory` function implemented below can then be used to obtain a UDF for the parsing method specified in the `pipeline_config` of the `00_config` notebook.
# MAGIC
# MAGIC ### Adding a New Parser
# MAGIC To add a new parser, follow these steps (refer to the provided example in the README):
# MAGIC - Ensure all required dependencies are installed in the next cells.
# MAGIC - Add another section in this notebook and implement the parsing function.
# MAGIC   - Name the function `parse_bytes_<method_name>`.
# MAGIC   - Ensure the output of the function complies with the `ParserReturnValue` class defined below to ensure compatibility with Spark UDFs.
# MAGIC - Add your new method to the `parser_factory` function defined below.
# MAGIC - For testing and development, include a simple testing function that loads the `test_data/test-document.pdf` file and asserts successful parsing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies
# MAGIC Each of the parser-specific dependencies are installed in separate cells to make it easier remove or add new parsers.

# COMMAND ----------

# DBTITLE 1,Dependencies for PyPdf Parser
# MAGIC %pip install -U -qqq pypdf==4.1.0

# COMMAND ----------

# DBTITLE 1,Dependencies for Azure Doc Intel Parser
# MAGIC %pip install -U -qqq azure-ai-documentintelligence==1.0.0b3

# COMMAND ----------

# DBTITLE 1,Dependencies for HTML to Markdown
# MAGIC %pip install -U -qqq markdownify==0.12.1

# COMMAND ----------

# DBTITLE 1,Dependencies for pyMuPDF
# MAGIC %pip install -U -qqq pymupdf4llm==0.0.5 pymupdf==1.24.5

# COMMAND ----------

# DBTITLE 1,PyPandocDocx
# MAGIC %pip install -U -qqq pypandoc_binary==1.13

# COMMAND ----------

# DBTITLE 1,Dependencies for UnstructuredIO
# MAGIC %pip install -U -qqq markdownify==0.12.1 "unstructured[local-inference, all-docs]==0.14.4" unstructured-client==0.22.0 pdfminer==20191125 nltk==3.8.1

# COMMAND ----------

# MAGIC %md ## Install system package dependencies

# COMMAND ----------

from typing import List
def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(
        1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    )

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e



# COMMAND ----------

# DBTITLE 1,PyPandocDocx
# install_apt_get_packages(['pandoc'])

# COMMAND ----------

# DBTITLE 1,Dependencies for UnstructuredIO
# install_apt_get_packages(["poppler-utils", "tesseract-ocr"])

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import io
from typing import List, Dict, Any, Tuple, Optional, TypedDict
import warnings
from functools import partial

import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, MapType

# COMMAND ----------

parser_debug_flag = False

# COMMAND ----------

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell defines the return type of the parser functions. The `ParserReturnValue` class serves as a type hint for the parser functions and must match the Spark schema, which is defined as the `returnType` of the UDF inside the `get_parser_udf` function.
# MAGIC
# MAGIC This function is used in the `parser_factory` below to convert any of the configured parser functions into a Spark UDF, which can be applied to the binary PDFs at scale.
# MAGIC
# MAGIC __Note__: The main output type is a `MapType`. This means that you can return any key-value pair from your parser in addition to your parsed text. For instance, this could include the total number of pages.
# MAGIC
# MAGIC If you require a more complex output format, such as individual pages along with their respective page numbers, you would need to choose a different schema. For example:
# MAGIC
# MAGIC
# MAGIC ```Python
# MAGIC class ParserReturnValue(TypedDict):
# MAGIC     OUTPUT_FIELD_NAME: List[Dict[str, str]]
# MAGIC     STATUS_FIELD_NAME: str
# MAGIC
# MAGIC returnType = StructType(
# MAGIC     [
# MAGIC         StructField(
# MAGIC             OUTPUT_FIELD_NAME,
# MAGIC             ArrayType(
# MAGIC                 StructType(
# MAGIC                     [
# MAGIC                         StructField("page_number", IntegerType()),
# MAGIC                         StructField("parsed_content", StringType()),
# MAGIC                     ]
# MAGIC                 )
# MAGIC             ),
# MAGIC             nullable=True,
# MAGIC         ),
# MAGIC         StructField(STATUS_FIELD_NAME, StringType(), nullable=True),
# MAGIC     ]
# MAGIC )
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Define Parser Output
OUTPUT_FIELD_NAME = "doc_parsed_contents"
STATUS_FIELD_NAME = "parser_status"

class ParserReturnValue(TypedDict):
    OUTPUT_FIELD_NAME: Dict[str, str]
    STATUS_FIELD_NAME: str


def get_parser_udf(parser_function):
    """Return a Spark UDF of the specified parsing function

    The return type matches the structure defined in the
    ParserReturnValue class
    """
    parser_udf = func.udf(
        parser_function,
        returnType=StructType(
            [
                StructField(
                    OUTPUT_FIELD_NAME,
                    MapType(StringType(), StringType()),
                    nullable=True,
                ),
                StructField(STATUS_FIELD_NAME, StringType(), nullable=True),
            ]
        ),
    )
    return parser_udf

# COMMAND ----------

# MAGIC %md
# MAGIC The `parser_factory` defined below assigns abbreviated names to the parsers and facilitates the configuration of different parsers through the `00_config` notebook. Any parser functions registered in the factory can be specified as the parser to be applied to the PDF binaries. The `parser_factory` function retrieves the specified parser from the configuration and returns the corresponding Spark UDF.

# COMMAND ----------

# DBTITLE 1,Define Parser Factory
def parser_factory(pipeline_config):

  # register all potential chunking methods and return respective UDF
  # different parser methods will require different parameters so we
  # use functools.partial to initialize them

  # Register all potential parsers and return as UDF
  if pipeline_config.get("parser").get("name") == "pypdf":
    return get_parser_udf(parse_bytes_pypdf)
  elif pipeline_config.get("parser").get("name") == "azure_doc_intelligence":
    return get_parser_udf(parse_bytes_adi)
  elif pipeline_config.get("parser").get("name") == "html_to_markdown":
    return get_parser_udf(parse_bytes_html_to_markdown) 
  elif pipeline_config.get("parser").get("name") == "pymupdf_markdown":
    return get_parser_udf(parse_bytes_pymupdfmarkdown) 
  elif pipeline_config.get("parser").get("name") == "pymupdf":
    return get_parser_udf(parse_bytes_pymupdf) 
  elif pipeline_config.get("parser").get("name") == "pypandocDocX":
    return get_parser_udf(parse_bytes_pypandocdocx) 
  elif pipeline_config.get("parser").get("name") == "unstructuredPDF":
    parser_config = pipeline_config.get("parser")
    return get_parser_udf(
                    partial(
                          parse_bytes_unstructuredPDF,
                          strategy = parser_config.get("config").get("strategy"),
                          hi_res_model_name = parser_config.get("config").get("hi_res_model_name"),
                          use_premium_features = parser_config.get("config").get("use_premium_features"),
                          api_url = parser_config.get("config").get("api_url"),
                          api_key = parser_config.get("config").get("api_key"),
                      )
            )
  elif pipeline_config.get("parser").get("name") == "unstructuredDocX":
    parser_config = pipeline_config.get("parser")
    return get_parser_udf(
                    partial(
                          parse_bytes_unstructuredDocX,
                          strategy = parser_config.get("config").get("strategy"),
                          hi_res_model_name = parser_config.get("config").get("hi_res_model_name"),
                          use_premium_features = parser_config.get("config").get("use_premium_features"),
                          api_url = parser_config.get("config").get("api_url"),
                          api_key = parser_config.get("config").get("api_key"),
                      )
            )
  elif pipeline_config.get("parser").get("name") == "unstructuredPPTX":
    parser_config = pipeline_config.get("parser")
    return get_parser_udf(
                    partial(
                          parse_bytes_unstructuredPPTX,
                          strategy = parser_config.get("config").get("strategy"),
                          hi_res_model_name = parser_config.get("config").get("hi_res_model_name"),
                          use_premium_features = parser_config.get("config").get("use_premium_features"),
                          api_url = parser_config.get("config").get("api_url"),
                          api_key = parser_config.get("config").get("api_key"),
                      )
            )
  elif pipeline_config.get("parser").get("name") == "json":
    parser_config = pipeline_config.get("parser")
    return get_parser_udf(
                      partial(
                            parse_bytes_json,
                            content_key = parser_config.get("config").get("content_key"),
                          )
                )

  
  else:
    raise ValueError(f"The {pipeline_config.get('parser')} parser is not implemented. Choose a different one from the ./parsers notebook")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Parsers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pypdf

# COMMAND ----------

from pypdf import PdfReader


def parse_bytes_pypdf(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)

        # TODO: How do we capture additional metadata such as page numbers
        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        output = {
            "num_pages": str(len(parsed_content)),
            "parsed_content": "\n".join(parsed_content),
        }

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    # TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"num_pages": "", "parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Pypdf ...

# COMMAND ----------

# with open("./test_data/test-document.pdf", "rb") as file:
#   file_bytes = file.read()
#   test_result_pypdf = parse_bytes_pypdf(file_bytes)

# # assert test_result_pypdf[STATUS_FIELD_NAME] == "SUCCESS"
# # assert test_result_pypdf['doc_parsed_contents']['num_pages'] == "27"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Document Intelligence
# MAGIC This is an Azure Service which requires authentication. You therefore have to provide:
# MAGIC - The endpoint url assigned to the `adi_endpoint` variable
# MAGIC - The authentication key assigned to the `adi_key` variable
# MAGIC
# MAGIC The [Azure documentation explains](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0#get-endpoint-url-and-keys) how to get both the endpoint and the key. It is recommended to store them in a [secret scope](https://docs.databricks.com/en/security/secrets/secret-scopes.html).
# MAGIC
# MAGIC *Note, the free tier will only parse 2 pages per request.*

# COMMAND ----------

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult


try:
    adi_endpoint = dbutils.secrets.get(scope="your_scope", key="afr_api_endpoint")
    adi_key = dbutils.secrets.get(scope="your_scope", key="afr_api_key")

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=adi_endpoint, credential=AzureKeyCredential(adi_key)
    )
except Exception as e:
    print("Failed to load Azure Doc Intelligence, make sure your secret is valid")
    print(e)


# TODO: Check different failure modes
def parse_bytes_adi(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=pdf,
            content_type="application/octet-stream",
            output_content_format="markdown",
        )
        result: AnalyzeResult = poller.result()
        
        output = {
            "num_pages": str(len(result.pages)),
            "parsed_content": result.content,
        }
        
        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    #TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"num_pages": "", "parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test Azure Doc Intelligence ...

# COMMAND ----------

# with open("./test_data/test-document.pdf", "rb") as file:
#   file_bytes = file.read()
#   test_result_adi = parse_bytes_adi(file_bytes)

# assert test_result_adi[STATUS_FIELD_NAME] == "SUCCESS"
# assert test_result_adi['doc_parsed_contents']['num_pages'] == "2"

# COMMAND ----------

# MAGIC %md
# MAGIC ## HTML to Markdown

# COMMAND ----------

from markdownify import markdownify as md
import markdownify
import re

def parse_bytes_html_to_markdown(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        html_content = raw_doc_contents_bytes.decode("utf-8")

        markdown_contents = md(str(html_content).strip(), heading_style=markdownify.ATX)
        markdown_stripped = re.sub(r"\n{3,}", "\n\n", markdown_contents.strip())

        output = {
            # "num_pages": str(len(parsed_content)),
            "parsed_content": markdown_stripped,
        }

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    # TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# with open("./test_data/test-document.html", "rb") as file:
#   file_bytes = file.read()
#   test_result_pypdf = parse_bytes_html_to_markdown(file_bytes)

# assert test_result_pypdf[STATUS_FIELD_NAME] == "SUCCESS"


# COMMAND ----------

# MAGIC %md
# MAGIC ## PyMuPdfMarkdown
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library, converting the output to Markdown.

# COMMAND ----------

import fitz
import pymupdf4llm

def parse_bytes_pymupdfmarkdown(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
        md_text = pymupdf4llm.to_markdown(pdf_doc)

        output = {
            # "num_pages": str(len(parsed_content)),
            "parsed_content": md_text.strip(),
        }

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    # TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# with open("./test_data/test-document.pdf", "rb") as file:
#   file_bytes = file.read()
#   test_result = parse_bytes_pymupdfmarkdown(file_bytes)

# assert test_result[STATUS_FIELD_NAME] == "SUCCESS"
# # assert test_result_adi['doc_parsed_contents']['num_pages'] == "2"

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyMuPdf
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library.

# COMMAND ----------

import fitz


def parse_bytes_pymupdf(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
        output_text = [page.get_text().strip() for page in pdf_doc]

        output = {
            "parsed_content": "\n".join(output_text)
        }

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    # TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# with open("./test_data/test-document.pdf", "rb") as file:
#   file_bytes = file.read()
#   test_result = parse_bytes_pymupdf(file_bytes)

# assert test_result[STATUS_FIELD_NAME] == "SUCCESS"
# # assert test_result_adi['doc_parsed_contents']['num_pages'] == "2"

# COMMAND ----------

# MAGIC %md
# MAGIC ## PyPandocDocx
# MAGIC
# MAGIC Parse a DocX file with Pandoc parser using the `pypandoc` library

# COMMAND ----------

import pypandoc
import tempfile

def parse_bytes_pypandocdocx(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(raw_doc_contents_bytes)
            temp_file_path = temp_file.name
            md = pypandoc.convert_file(temp_file_path, "markdown", format="docx")

            output = {
                "parsed_content": md.strip()
            }

            return {
                OUTPUT_FIELD_NAME: output,
                STATUS_FIELD_NAME: "SUCCESS",
            }
    # TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: {"parsed_content": ""},
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# with open("./test_data/test-document.docx", "rb") as file:
#   file_bytes = file.read()
#   test_result = parse_bytes_pypandocdocx(file_bytes)

# assert test_result[STATUS_FIELD_NAME] == "SUCCESS"
# # assert test_result_adi['doc_parsed_contents']['num_pages'] == "2"

# COMMAND ----------

# MAGIC %md
# MAGIC ## UnstructuredPDF
# MAGIC
# MAGIC
# MAGIC Parse a PDF file with `unstructured` library. The Unstructured PDF parser, which enables both free local parsing as well as premium API-based parsing of .pdf files. Defaults to using the `hi_res` strategy with the `yolox` model.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC  """ The Unstructured PDF parser, which enables both free local parsing as well as premium API-based parsing of .pdf files.
# MAGIC
# MAGIC     Parameters:
# MAGIC     - strategy (str): The strategy to use for parsing the PDF document. Options include:
# MAGIC         - "ocr_only": Runs the document through Tesseract for OCR and then processes the raw text. Recommended for documents with multiple columns that do not have extractable text. Falls back to "fast" if Tesseract is not available and the document has extractable text.
# MAGIC         - "fast": Extracts text using pdfminer and processes the raw text. Recommended for most cases where the PDF has extractable text. Falls back to "ocr_only" if the text is not extractable.
# MAGIC         - "hi_res": Identifies the layout of the document using a specified model (e.g., detectron2_onnx). Uses the document layout to gain additional information about document elements. Recommended if your use case is highly sensitive to correct classifications for document elements. Falls back to "ocr_only" if the specified model is not available.
# MAGIC         The default strategy is "hi_res".
# MAGIC     - hi_res_model_name (str): The name of the model to use for the "hi_res" strategy. Options include:
# MAGIC         - "detectron2_onnx": A Computer Vision model by Facebook AI that provides object detection and segmentation algorithms with ONNX Runtime. It is the fastest model for the "hi_res" strategy.
# MAGIC         - "yolox": A single-stage real-time object detector that modifies YOLOv3 with a DarkNet53 backbone.
# MAGIC         - "yolox_quantized": Runs faster than YoloX and its speed is closer to Detectron2.
# MAGIC         The default model is "yolox".
# MAGIC     - use_premium_features (bool): Whether to use premium, proprietary models and features for document parsing. These models may offer better accuracy or additional features compared to open-source models, but require an API key and endpoint URL for access. Set to `True` to enable the use of premium models. The default is `False`.
# MAGIC         The default setting is False.
# MAGIC     - api_key (str): The API key required to access the premium parsing engine. This is only needed if `use_premium_models` is set to `True`. This key authenticates the requests to the premium API service.
# MAGIC         The default setting is "".
# MAGIC     - api_url (str): The URL of the API endpoint for accessing premium parsing engine. This should be provided if `use_premium_models` is set to `True`. For Unstructured-hosted SaaS API, the format should be https://{{UNSTRUCT_SAAS_API_TENANT_ID}}.api.unstructuredapp.io/ and https://api.unstructured.io/general/v0/general/ for the Unstructured-hosted, capped-usage, free API.
# MAGIC         The default setting is "".
# MAGIC     """ 

# COMMAND ----------



def parse_bytes_unstructuredPDF(
    raw_doc_contents_bytes: bytes,
    strategy:str = "hi_res",       #Strategy to use for parsing. Options: "hi_res", "ocr_only", "fast"
    hi_res_model_name:str="yolox", #hi_res model name. Options  "yolox", "yolox_quantized", "detectron2_onnx"
    doc_name: str = "filename_unavailable.pdf", # TODO: can the "doc_uri" param be passed into the parse_bytes method?
    use_premium_features: bool|None = None,     # optional; allows user to toggle/on premium features on a per call basis
    api_key:str="", #optional; needed for premium features
    api_url:str="",  #optional; needed for premium features
    **kwargs
) -> ParserReturnValue:
    from unstructured.partition.pdf import partition_pdf
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared
    from unstructured_client.models.errors import SDKError
    from unstructured.staging.base import elements_from_json
    import json
    import io
    from markdownify import markdownify as md
    
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        #If args are None, then set them to default values
        if (strategy is None): strategy = "hi_res" 
        if (hi_res_model_name is None ): hi_res_model_name = "yolox"
        if (doc_name is None ): doc_name = "filename_unavailable.pdf"

        parsing_base_config = {
            "strategy": strategy, # default to use ``hi_res`` strategy
            "hi_res_model_name": hi_res_model_name,
            "skip_infer_table_types": [],
            "extract_image_block_types": ["Image", "Table"], # optional; useful for relaying tougher images/tables to vision models
        }

        api_config = {
            **parsing_base_config,
            "files": shared.Files(
                content=raw_doc_contents_bytes,
                file_name=doc_name
            ),
            "split_pdf_page": True, # Runs concurrently
            }

        local_config = {
            **parsing_base_config,
            "file": pdf,
            "infer_table_structure": True,
            "extract_image_block_to_payload": True, # optional
            }
        # The use_premium_features flag routes requests to an Unstructured-hosted or client-hosted Marketplace API (Free, SaaS, or Marketplace VPC)
        if use_premium_features:
            client = UnstructuredClient(
                server_url=api_url,
                api_key_auth=api_key,
            )
            try: 
                req = shared.PartitionParameters(**api_config)
                resp = client.general.partition(req)
                document_sections = elements_from_json(text=json.dumps(resp.elements))
            except Exception as e:
                raise SDKError(f"Error parsing document {doc_name} via the premium Unstructured API: {e}")
        else:
            try: 
                document_sections = partition_pdf(**local_config, **kwargs)
            except Exception as e:
                raise SDKError(f"Error parsing document {doc_name} via the Unstructured open source library: {e}")
        text_content = ""
        for section in document_sections:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text
        
        output = {
            "parsed_content": text_content,
        }
        value_to_return = output

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    #TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: [{"page_number": None, "parsed_content": None}],
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }
    

# COMMAND ----------

parser_debug_flag = False
if parser_debug_flag:
  # debug code
  with open(
        "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pdf_1only/2305.05176.pdf",
        "rb",
  ) as file:
        file_bytes = file.read()
        test_result = parse_bytes_unstructuredPDF(file_bytes)
        print(test_result)
        assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

parser_debug_flag = False
if parser_debug_flag:
  # debug code
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pdf/2305.05176.pdf",
      "rb",
  ) as file:
      file_bytes = file.read()
      test_result = parse_bytes_unstructuredPDF(file_bytes,use_premium_features=True,api_key=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_key"),api_url=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_url") )
      print(test_result)
      assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

parser_debug_flag = False
# testing with configs passed to the parser
if parser_debug_flag:
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pdf/2305.05176.pdf",
      "rb",
  ) as file:
      parser_udf = parser_factory(configurations.get("poc").get("pipeline_config"))
      file_bytes = file.read()
      from pyspark.sql.types import BinaryType
      df = spark.createDataFrame([(file_bytes,)], ["content"])
      df = df.withColumn("content", df["content"].cast(BinaryType())). withColumn("parsing", parser_udf("content"))
      display(df)
parser_debug_flag = False

# COMMAND ----------

# MAGIC %md ## UnstructuredDocX
# MAGIC
# MAGIC Parse a DocX file with the `unstructured` library. The Unstructured DocX parser, which enables both free local parsing as well as premium API-based parsing of .docx files.

# COMMAND ----------

# MAGIC %md
# MAGIC """ The Unstructured DocX parser, which enables both free local parsing as well as premium API-based parsing of .docx files.
# MAGIC
# MAGIC     Parameters:
# MAGIC     - strategy (str): The strategy to use for parsing the DocX document. Options include:
# MAGIC         - "ocr_only": Runs the document through Tesseract for OCR and then processes the raw text. Recommended for documents with multiple columns that do not have extractable text. Falls back to "fast" if Tesseract is not available and the document has extractable text.
# MAGIC         - "fast": Extracts text and processes the raw text. Recommended for most cases where the DocX has extractable text. Falls back to "ocr_only" if the text is not extractable.
# MAGIC         - "hi_res": Identifies the layout of the document using a specified model (e.g., detectron2_onnx). Uses the document layout to gain additional information about document elements. Recommended if your use case is highly sensitive to correct classifications for document elements. Falls back to "ocr_only" if the specified model is not available.
# MAGIC         The default strategy is "hi_res".
# MAGIC     - hi_res_model_name (str): The name of the model to use for the "hi_res" strategy. Options include:
# MAGIC         - "detectron2_onnx": A Computer Vision model by Facebook AI that provides object detection and segmentation algorithms with ONNX Runtime. It is the fastest model for the "hi_res" strategy.
# MAGIC         - "yolox": A single-stage real-time object detector that modifies YOLOv3 with a DarkNet53 backbone.
# MAGIC         - "yolox_quantized": Runs faster than YoloX and its speed is closer to Detectron2.
# MAGIC         The default model is "yolox".
# MAGIC     - use_premium_features (bool): Whether to use premium, proprietary models and features for document parsing. These models may offer better accuracy or additional features compared to open-source models, but require an API key and endpoint URL for access. Set to `True` to enable the use of premium models. The default is `False`.
# MAGIC         The default setting is False.
# MAGIC     - api_key (str): The API key required to access the premium parsing engine. This is only needed if `use_premium_models` is set to `True`. This key authenticates the requests to the premium API service.
# MAGIC         The default setting is "".
# MAGIC     - api_url (str): The URL of the API endpoint for accessing premium parsing engine. This should be provided if `use_premium_models` is set to `True`. For Unstructured-hosted SaaS API, the format should be https://{{UNSTRUCT_SAAS_API_TENANT_ID}}.api.unstructuredapp.io/ and https://api.unstructured.io/general/v0/general/ for the Unstructured-hosted, capped-usage, free API.
# MAGIC         The default setting is "".
# MAGIC     """

# COMMAND ----------



def parse_bytes_unstructuredDocX(
    raw_doc_contents_bytes: bytes,
    strategy:str = "hi_res",       #Strategy to use for parsing. Options: "hi_res", "ocr_only", "fast"
    hi_res_model_name:str="yolox", #hi_res model name. Options  "yolox", "yolox_quantized", "detectron2_onnx"
    doc_name: str = "filename_unavailable.docx", # TODO: can the "doc_uri" param be passed into the parse_bytes method?
    use_premium_features: bool|None = None,     # optional; allows user to toggle/on premium features on a per call basis
    api_key:str="", #optional; needed for premium features
    api_url:str="",  #optional; needed for premium features
    **kwargs
) -> ParserReturnValue:
    from unstructured.partition.docx import partition_docx
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared
    from unstructured_client.models.errors import SDKError
    from unstructured.staging.base import elements_from_json
    import json        
    import io
    from markdownify import markdownify as md
    
    try:
        reconstructed_file = io.BytesIO(raw_doc_contents_bytes)
        #If args are None, then set them to default values
        if (strategy is None): strategy = "hi_res" 
        if (hi_res_model_name is None ): hi_res_model_name = "yolox"
        if (doc_name is None ): doc_name = "filename_unavailable.docx"
        
        parsing_base_config = {
            "strategy": strategy, # mandatory to use ``hi_res`` strategy
            "hi_res_model_name": hi_res_model_name,
            "skip_infer_table_types": [],
        }

        api_config = {
            **parsing_base_config,
            "files": shared.Files(
                content=raw_doc_contents_bytes,
                file_name=doc_name
            ),
            "extract_image_block_types": ["Image", "Table"], # optional; useful for relaying tougher images/tables to vision models; only available as a premium feature
            }

        local_config = {
            **parsing_base_config,
            "file": reconstructed_file,
            "source_format": "docx",
            "infer_table_structure": True,
            "extract_image_block_to_payload": True, # optional
            }
        # The use_premium_features flag routes requests to an Unstructured-hosted or client-hosted Marketplace API (Free, SaaS, or Marketplace VPC)
        if use_premium_features:
            try: 
                client = UnstructuredClient(
                    server_url=api_url,
                    api_key_auth=api_key,
                )
                req = shared.PartitionParameters(**api_config)
                resp = client.general.partition(req)
                document_sections = elements_from_json(text=json.dumps(resp.elements))
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the premium Unstructured API: {e}")
        else:
            try: 
                document_sections = partition_docx(**local_config, **kwargs)
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the Unstructured open source library: {e}")
        text_content = ""
        for section in document_sections:
            # Tables are parsed seperatly, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text
        
        output = {
            "parsed_content": text_content,
        }
        value_to_return = output

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    #TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: [{"page_number": None, "parsed_content": None}],
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }
    

# COMMAND ----------

#parser_debug_flag = False
if parser_debug_flag:
  # debug code
  with open(
        "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/docx/Freddie Mac Loan Agreement.docx",
        "rb",
  ) as file:
        file_bytes = file.read()
        test_result = parse_bytes_unstructuredDocX(file_bytes)
        print(test_result)
        assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

#parser_debug_flag = False
if parser_debug_flag:
  # debug code
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/docx/Freddie Mac Loan Agreement.docx",
      "rb",
  ) as file:
      file_bytes = file.read()
      test_result = parse_bytes_unstructuredDocX(file_bytes,use_premium_features=True,api_key=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_key"),api_url=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_url") )
      print(test_result)
      assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

#parser_debug_flag = True
# testing with configs passed to the parser
if parser_debug_flag:
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/docx/Freddie Mac Loan Agreement.docx",
      "rb",
  ) as file:
      parser_udf = parser_factory(configurations.get("poc").get("pipeline_config"))
      file_bytes = file.read()
      from pyspark.sql.types import BinaryType
      df = spark.createDataFrame([(file_bytes,)], ["content"])
      df = df.withColumn("content", df["content"].cast(BinaryType())). withColumn("parsing", parser_udf("content"))
      display(df)
#parser_debug_flag = False

# COMMAND ----------

# MAGIC %md ## UnstructuredPPTX
# MAGIC
# MAGIC Parse a PPTX file with the `unstructured` library. The Unstructured PPTX parser, which enables both free local parsing as well as premium API-based parsing of .pptx files.

# COMMAND ----------

# MAGIC %md
# MAGIC """ The Unstructured PPTX parser, which enables both free local parsing as well as premium API-based parsing of .pptx files.
# MAGIC
# MAGIC     Parameters:
# MAGIC     - strategy (str): The strategy to use for parsing the PPTX document. Options include:
# MAGIC         - "ocr_only": Runs the document through Tesseract for OCR and then processes the raw text. Recommended for documents with multiple columns that do not have extractable text. Falls back to "fast" if Tesseract is not available and the document has extractable text.
# MAGIC         - "fast": Extracts text and processes the raw text. Recommended for most cases where the PPTX has extractable text. Falls back to "ocr_only" if the text is not extractable.
# MAGIC         - "hi_res": Identifies the layout of the document using a specified model (e.g., detectron2_onnx). Uses the document layout to gain additional information about document elements. Recommended if your use case is highly sensitive to correct classifications for document elements. Falls back to "ocr_only" if the specified model is not available.
# MAGIC         The default strategy is "hi_res".
# MAGIC     - hi_res_model_name (str): The name of the model to use for the "hi_res" strategy. Options include:
# MAGIC         - "detectron2_onnx": A Computer Vision model by Facebook AI that provides object detection and segmentation algorithms with ONNX Runtime. It is the fastest model for the "hi_res" strategy.
# MAGIC         - "yolox": A single-stage real-time object detector that modifies YOLOv3 with a DarkNet53 backbone.
# MAGIC         - "yolox_quantized": Runs faster than YoloX and its speed is closer to Detectron2.
# MAGIC         The default model is "yolox".
# MAGIC     - use_premium_features (bool): Whether to use premium, proprietary models and features for document parsing. These models may offer better accuracy or additional features compared to open-source models, but require an API key and endpoint URL for access. Set to `True` to enable the use of premium models. The default is `False`.
# MAGIC         The default setting is False.
# MAGIC     - api_key (str): The API key required to access the premium parsing engine. This is only needed if `use_premium_models` is set to `True`. This key authenticates the requests to the premium API service.
# MAGIC         The default setting is "".
# MAGIC     - api_url (str): The URL of the API endpoint for accessing premium parsing engine. This should be provided if `use_premium_models` is set to `True`. For Unstructured-hosted SaaS API, the format should be https://{{UNSTRUCT_SAAS_API_TENANT_ID}}.api.unstructuredapp.io/ and https://api.unstructured.io/general/v0/general/ for the Unstructured-hosted, capped-usage, free API.
# MAGIC         The default setting is "".
# MAGIC     """    

# COMMAND ----------



def parse_bytes_unstructuredPPTX(
    raw_doc_contents_bytes: bytes,
    strategy:str = "hi_res",       #Strategy to use for parsing. Options: "hi_res", "ocr_only", "fast"
    hi_res_model_name:str="yolox", #hi_res model name. Options  "yolox", "yolox_quantized", "detectron2_onnx"
    doc_name: str = "filename_unavailable.pptx", # TODO: can the "doc_uri" param be passed into the parse_bytes method?
    use_premium_features: bool|None = None,     # optional; allows user to toggle/on premium features on a per call basis
    api_key:str="", #optional; needed for premium features
    api_url:str="",  #optional; needed for premium features
    **kwargs
) -> ParserReturnValue:
    from unstructured.partition.pptx import partition_pptx
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared
    from unstructured_client.models.errors import SDKError
    from unstructured.staging.base import elements_from_json
    import json        
    import io
    from markdownify import markdownify as md
    
    try:
        reconstructed_file = io.BytesIO(raw_doc_contents_bytes)
        #If args are None, then set them to default values
        if (strategy is None): strategy = "hi_res" 
        if (hi_res_model_name is None ): hi_res_model_name = "yolox"
        if (doc_name is None ): doc_name = "filename_unavailable.pptx"

        parsing_base_config = {
            "strategy": strategy, # mandatory to use ``hi_res`` strategy
            "hi_res_model_name": hi_res_model_name,
            "skip_infer_table_types": [], # file types to skip
            "extract_image_block_types": ["Image", "Table"], # optional
        }
        api_config = {
            **parsing_base_config,
            "files": shared.Files(
                content=raw_doc_contents_bytes,
                file_name=doc_name
            ),
            }

        local_config = {
            **parsing_base_config,
            "file": reconstructed_file,
            "source_format": "docx",
            "infer_table_structure": True,
            "extract_image_block_to_payload": True, # optional
            }
        # The use_premium_features flag routes requests to an Unstructured-hosted or client-hosted Marketplace API (Free, SaaS, or Marketplace VPC)
        
        if use_premium_features:
            try: 
                client = UnstructuredClient(
                    server_url=api_url,
                    api_key_auth=api_key,
                )
                req = shared.PartitionParameters(**api_config)
                resp = client.general.partition(req)
                document_sections = elements_from_json(text=json.dumps(resp.elements))
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the premium Unstructured API: {e}")
        else:
            try: 
                document_sections = partition_pptx(**local_config, **kwargs)
            except Exception as e:
                raise SDKError(f"Error parsing document doc_name via the Unstructured open source library: {e}")
        text_content = ""
        for section in document_sections:
            # Tables are parsed seperately, add a \n to give the chunker a hint to split well.
            if section.category == "Table":
                if section.metadata is not None:
                    if section.metadata.text_as_html is not None:
                        # convert table to markdown
                        text_content += "\n" + md(section.metadata.text_as_html) + "\n"
                    else:
                        text_content += " " + section.text
                else:
                    text_content += " " + section.text
            # Other content often has too-aggresive splitting, merge the content
            else:
                text_content += " " + section.text
        
        output = {
            "parsed_content": text_content,
        }
        value_to_return = output

        return {
            OUTPUT_FIELD_NAME: output,
            STATUS_FIELD_NAME: "SUCCESS",
        }
    #TODO: Be more specific about the exception
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_FIELD_NAME: [{"page_number": None, "parsed_content": None}],
            STATUS_FIELD_NAME: f"ERROR: {e}",
        }
    

# COMMAND ----------

#parser_debug_flag = True
if parser_debug_flag:
  # debug code
  with open(
        "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pptx/Unit -1 Introduction to Strategies.pptx",
        "rb",
  ) as file:
        file_bytes = file.read()
        test_result = parse_bytes_unstructuredPPTX(file_bytes)
        print(test_result)
        assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

#parser_debug_flag = True
if parser_debug_flag:
  # debug code
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pptx/Unit -1 Introduction to Strategies.pptx",
      "rb",
  ) as file:
      file_bytes = file.read()
      test_result = parse_bytes_unstructuredPPTX(file_bytes,use_premium_features=True,api_key=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_key"),api_url=dbutils.secrets.get(scope="rag_on_databricks_pk", key="unstructured_io_api_url") )
      print(test_result)
      assert test_result[STATUS_FIELD_NAME] == "SUCCESS"

# COMMAND ----------

#parser_debug_flag = True
# testing with configs passed to the parser
if parser_debug_flag:
  with open(
      "/Volumes/main/rag_studio_demo_prasadkona/demo_docs/pptx/Unit -1 Introduction to Strategies.pptx",
      "rb",
  ) as file:
      parser_udf = parser_factory(configurations.get("poc").get("pipeline_config"))
      file_bytes = file.read()
      from pyspark.sql.types import BinaryType
      df = spark.createDataFrame([(file_bytes,)], ["content"])
      df = df.withColumn("content", df["content"].cast(BinaryType())). withColumn("parsing", parser_udf("content"))
      display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC # JSON files
# MAGIC
# MAGIC `content_key`: JSON key containing the string content that should be chunked
# MAGIC
# MAGIC All other keys will be passed through as-is

# COMMAND ----------

import json

def parse_bytes_json(
    raw_doc_contents_bytes: bytes,
    content_key: str
) -> ParserReturnValue:

    try:
        # Decode the raw bytes from Spark
        json_str = raw_doc_contents_bytes.decode("utf-8")
        # Load the JSON contained in the bytes
        json_data = json.loads(json_str)
        
        # Load the known key to `parsed_content`
        json_data['parsed_content'] = json_data[content_key]
        
        # Remove that key
        del json_data[content_key]
        
        output = json_data
        status = "SUCCESS"

    except json.JSONDecodeError as e:
        status = f"JSON decoding failed: {e}"
        output = {
            "parsed_content": "",
        }
        warnings.warn(status)
    except UnicodeDecodeError as e:
        status = f"Unicode decoding failed: {e}"
        output = {
            "parsed_content": "",
        }
        warnings.warn(status)
    except Exception as e:
        status = f"An unexpected error occurred: {e}"
        output = {
            "parsed_content": "",
        }

    return {
        OUTPUT_FIELD_NAME: output,
        STATUS_FIELD_NAME: status,
    }



# COMMAND ----------

# debug code
# parser_debug_flag=True
if parser_debug_flag:
    with open(
        "./test_data/test-document.json",
        "rb",
    ) as file:
        file_bytes = file.read()
        test_result = parse_bytes_json(file_bytes, content_key="html_content")
        print(test_result)
        assert test_result[STATUS_FIELD_NAME] == "SUCCESS"
        test_result['doc_parsed_contents'].keys()
