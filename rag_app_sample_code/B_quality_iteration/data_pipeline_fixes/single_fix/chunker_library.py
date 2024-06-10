# Databricks notebook source
# MAGIC %md
# MAGIC ## Chunker Library
# MAGIC This notebook implements a library of chunking tools. This notebook is supposed to be `%run` as the first cell inside the `03_chunk_docs` notebook. The `chunker_factory` function implemented below can then be used to get a udf for the chunking method specified in the `pipeline_config` of the `00_config` notebook.
# MAGIC
# MAGIC ### To add a new chunker
# MAGIC If you want to add a new chunker you have to follow a few simple steps:
# MAGIC - Ensure all required dependencies are installed in the next cells.
# MAGIC - Add another section in this notebook and implement the parsing function
# MAGIC     - Name the function `chunk_parsed_content_<method_name>`
# MAGIC     - Ensure the output of the function complies with the `ChunkerReturnValue` class defined below to ensure compatibility with Spark UDFs.
# MAGIC - Add your new method to the `chunker_factory` function defined below
# MAGIC - For testing and development, include a simple testing function that chunks a pre-defined text asserts successful chunking.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install dependencies
# MAGIC Each of the chunker-specific dependencies are installed in separate cells to make it easier remove or add new chunkers.

# COMMAND ----------

# DBTITLE 1,Dependencies for langchain based split chunker
# MAGIC %pip install -U -qqq transformers==4.41.1 torch==2.3.0 tiktoken==0.7.0 langchain-text-splitters==0.2.0

# COMMAND ----------

# DBTITLE 1,Dependencies for langchain markdown splitter
# MAGIC %pip install -U -qqq langchain-text-splitters==0.2.0

# COMMAND ----------

# DBTITLE 1,Semantic Chunker
# MAGIC %pip install -U -qqq nltk==3.8.1 transformers==4.41.2 torch==2.3.0 scikit-learn==1.5.0

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shared Imports & Utilities

# COMMAND ----------

from functools import partial
from typing import TypedDict, Dict
import warnings

import pyspark.sql.functions as func
from pyspark.sql.types import StructType, StringType, StructField, ArrayType

# COMMAND ----------

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell defines the return type of the chunker functions. The `ChunkerReturnValue` class serves as a type hint for the chunker functions and must match the Spark schema, which is defined as the `returnType` of the UDF inside the `get_chunker_udf` function.
# MAGIC
# MAGIC __Note__: We are returning a list of strings (`ArrayType(StringType())` in Spark) because each individual parsed text will be divided into many smaller chunks.
# MAGIC
# MAGIC This function is used in the `chunker_factory` below to convert any of the configured chunker functions into a Spark UDF, which can then be applied to the parsed documents at scale.

# COMMAND ----------

# DBTITLE 1,Define Chunker Output
OUTPUT_COLUMN_NAME = "chunked_text"
STATUS_COLUMN_NAME = "chunker_status"


class ChunkerReturnValue(TypedDict):
    OUTPUT_COLUMN_NAME: str
    STATUS_COLUMN_NAME: str


def get_chunker_udf(chunking_function):
    chunker_udf = func.udf(
        chunking_function,
        returnType=StructType(
            [
                StructField(OUTPUT_COLUMN_NAME, ArrayType(StringType()), nullable=True),
                StructField(STATUS_COLUMN_NAME, StringType(), nullable=True),
            ]
        ),
    )
    return chunker_udf

# COMMAND ----------

# MAGIC %md
# MAGIC The `chunker_factory` defined below assigns abbreviated names to the chunkers and facilitates the configuration of different chunkers through the `00_config` notebook. Any chunker functions registered in the factory can be specified as the chunker to be applied to the parsed documents. The `chunker_factory` function retrieves the specified chunker from the configuration and returns the corresponding Spark UDF.
# MAGIC
# MAGIC __Note__: If your add a chunker that requires configuration parameters in addition to the parsed text itself you can follow the example demonstrated here. It uses [functools' partial](https://docs.python.org/3/library/functools.html#functools.partial) to pre-configure the configuration parameters. This is necessary because UDFs only accept one input parameter, which will be the parsed text in our case.

# COMMAND ----------

# DBTITLE 1,Define Chunker Factory
def chunker_factory(pipeline_config, embedding_config):
    # register all potential chunking methods and return respective UDF
    # different chunking methods will require different parameters so we
    # use functools.partial to initialize them
    chunker_conf = pipeline_config.get("chunker")
    if chunker_conf.get("name") == "langchain_recursive_char":
        return get_chunker_udf(
            partial(
                chunk_parsed_content_langrecchar,
                chunk_size=chunker_conf.get("config").get("chunk_size_tokens"),
                chunk_overlap=chunker_conf.get("config").get("chunk_overlap_tokens"),
                embedding_config=embedding_config,
            )
        )
    elif chunker_conf.get("name") == "langchain_markdown_headers":
        return get_chunker_udf(
            partial(
                chunk_parsed_content_markdownheaders,
                include_headers_in_chunks=chunker_conf.get("config").get("include_headers_in_chunks"),
            )
        )
    elif chunker_conf.get("name") == "semantic":
        max_chunk_size = chunker_conf.get("config").get("max_chunk_size", 500)
        split_distance_percentile = chunker_conf.get("config").get("split_distance_percentile", .95)
        min_sentences = chunker_conf.get("config").get("min_sentences", 3)

        chunker = SemanticTextChunker(max_chunk_size, split_distance_percentile, min_sentences)
        return chunker.get_chunker_udf()
    else:
        raise ValueError(f"The {chunker_conf.get('name')} chunker is not implemented. Choose a different one from the ./chunkers notebook")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Langchain Recursive Character Split

# COMMAND ----------

import tiktoken
from transformers import AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_parsed_content_langrecchar(
    doc_parser_output: Dict[str, str], chunk_size: int, chunk_overlap: int, embedding_config
) -> ChunkerReturnValue:
    try:
        if (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "hugging_face"
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif (
            embedding_config.get("embedding_tokenizer").get("tokenizer_source")
            == "tiktoken"
        ):
            tokenizer = tiktoken.encoding_for_model(
                embedding_config.get("embedding_tokenizer").get("tokenizer_model_name")
            )

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        parsed_content = doc_parser_output.get("parsed_content")

        chunks = text_splitter.split_text(parsed_content)
        return {
            OUTPUT_COLUMN_NAME: [doc for doc in chunks],
            STATUS_COLUMN_NAME: "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_COLUMN_NAME: [],
            STATUS_COLUMN_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test langchain's recursive character split chunker

# COMMAND ----------

test_doc = """This page explains what a feature store is and what benefits it provides, and the specific advantages of Databricks Feature Store.

A feature store is a centralized repository that enables data scientists to find and share features and also ensures that the same code used to compute the feature values is used for model training and inference.

Machine learning uses existing data to build a model to predict future outcomes. In almost all cases, the raw data requires preprocessing and transformation before it can be used to build a model. This process is called feature engineering, and the outputs of this process are called features - the building blocks of the model.

Developing features is complex and time-consuming. An additional complication is that for machine learning, feature calculations need to be done for model training, and then again when the model is used to make predictions. These implementations may not be done by the same team or using the same code environment, which can lead to delays and errors. Also, different teams in an organization will often have similar feature needs but may not be aware of work that other teams have done. A feature store is designed to address these problems."""

embedding_config_test = {
    "embedding_endpoint_name": "databricks-bge-large-en",
    "embedding_tokenizer": {
        "tokenizer_model_name": "BAAI/bge-large-en-v1.5",
        "tokenizer_source": "hugging_face",
    },
}
chunk_parsed_content_langrecchar(test_doc, 475, 50, embedding_config_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LangChain Markdown header splitter

# COMMAND ----------

from langchain_text_splitters import  MarkdownHeaderTextSplitter

def chunk_parsed_content_markdownheaders(
    doc_parser_output: Dict[str, str], include_headers_in_chunks: bool
) -> ChunkerReturnValue:
    try:

        parsed_content = doc_parser_output.get("parsed_content")
        section_headers = doc_parser_output.get("section_headers")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = markdown_splitter.split_text(parsed_content)
        formatted_chunks = []

        if include_headers_in_chunks:
          for chunk in chunks:
            out_text = ''
            for (header_name, header_content) in chunk.metadata.items():
              out_text += f"{header_name}: {header_content}\n" 
              out_text += chunk.page_content
            formatted_chunks.append(out_text)
        else:
          for chunk in chunks:
            formatted_chunks.append(chunk.page_content)

        return {
            OUTPUT_COLUMN_NAME: formatted_chunks,
            STATUS_COLUMN_NAME: "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            OUTPUT_COLUMN_NAME: [],
            STATUS_COLUMN_NAME: f"ERROR: {e}",
        }

# COMMAND ----------

test_doc = {"parsed_content": """# Hello world 
This page explains what a feature store is and what benefits it provides, and the specific advantages of Databricks Feature Store.

A feature store is a centralized repository that enables data scientists to find and share features and also ensures that the same code used to compute the feature values is used for model training and inference.

Machine learning uses existing data to build a model to predict future outcomes. In almost all cases, the raw data requires preprocessing and transformation before it can be used to build a model. This process is called feature engineering, and the outputs of this process are called features - the building blocks of the model.

## Hello earth
Developing features is complex and time-consuming. An additional complication is that for machine learning, feature calculations need to be done for model training, and then again when the model is used to make predictions. These implementations may not be done by the same team or using the same code environment, which can lead to delays and errors. Also, different teams in an organization will often have similar feature needs but may not be aware of work that other teams have done. A feature store is designed to address these problems."""}


chunk_parsed_content_markdownheaders(test_doc, False)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Chunker
# MAGIC
# MAGIC
# MAGIC Rather than assigning a fixed chunk size, the semantic chunker uses the semantic similarity of sentence embeddings to determine where to break up the text.
# MAGIC
# MAGIC This requires embeddings to be computed twice, so it is slower and more costly compared to fixed size chunking strategies. Some cases when you might consider using this: 
# MAGIC
# MAGIC - Dense texts such as journal articles that contain many discrete concepts in close proximity. 
# MAGIC - Narrative texts such as news articles or novels that don't contain much markup / document headings.
# MAGIC
# MAGIC `max_chunk_size`: the number of tokens for max length of a single chunk
# MAGIC `split_distance_percentile`: consecutive sentences that have a cosine distance in above this percentile will be chunked
# MAGIC `min_sentences`: The smallest grouping of sentences, if set to 1 splitting can be sensitive/noisy
# MAGIC
# MAGIC By default, this chunker uses the `databricks-gte-large` Embedding model, but you can replace this with any other Model Serving endpoint supporting the `/llm/v1/embeddings` signature such as an OpenAI External Model.  If you change the embedding model, make sure to update the tokenizer to match.

# COMMAND ----------

import nltk
# from langchain.embeddings import DatabricksEmbeddings
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField

#TODO: Replace with GTE
# The tokenizer must match the embedding model.
EMBEDDING_ENDPOINT_NAME = "databricks-bge-large-en"
HF_TOKENIZER_NAME = "BAAI/bge-large-en-v1.5"

async def batch_serving_endpoint_embedding(texts):
    import time
    import httpx
    import traceback
    import asyncio
    from mlflow.utils.databricks_utils import get_databricks_host_creds
    from tenacity import (
        retry,
        stop_after_attempt,
        retry_if_exception,
        wait_random_exponential,
    )

    config_timeout = 300
    config_max_retries_other = 5
    config_max_retries_backpressure = 5
    config_logging_interval = 3
    config_concurrency = 20
    config_endpoint = EMBEDDING_ENDPOINT_NAME

    def is_backpressure(error: httpx.HTTPStatusError):
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            return error.response.status_code in (429, 503)


    def is_other_error(error: httpx.HTTPStatusError):
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            return error.response.status_code != 503 and (
                error.response.status_code >= 500 or error.response.status_code == 408
            )


    class AsyncChatClient:
        def __init__(self):
            self.client = httpx.AsyncClient(timeout=config_timeout)
            self.endpoint = config_endpoint
            # self.prompt = config_prompt
            # self.request_params = config_request_params
            # print(f"[AsyncChatClient] prompt: {self.prompt}")
            # print(f"[AsyncChatClient] request parameters: {self.request_params}")

        @retry(
            retry=retry_if_exception(is_other_error),
            stop=stop_after_attempt(config_max_retries_other),
            wait=wait_random_exponential(multiplier=1, max=20),
        )
        @retry(
            retry=retry_if_exception(is_backpressure),
            stop=stop_after_attempt(config_max_retries_backpressure),
            wait=wait_random_exponential(multiplier=1, max=20),
        )
        async def predict(self, text):
            credencials = get_databricks_host_creds("databricks")
            url = f"{credencials.host}/serving-endpoints/{self.endpoint}/invocations"
            headers = {
                "Authorization": f"Bearer {credencials.token}",
                "Content-Type": "application/json",
            }

            # messages = []
            # if self.prompt:
            #     messages.append({"role": "user", "content": self.prompt + str(text)})
            # else:
            #     messages.append({"role": "user", "content": str(text)})

            response = await self.client.post(
                url=url,
                headers=headers,
                json={"input": text},
            )
            response.raise_for_status()
            response = response.json()
            return (
                response["data"][0]["embedding"]
            )

        async def close(self):
            await self.client.aclose()

    class AsyncCounter:
        def __init__(self):
            self.value = 0

        async def increment(self):
            self.value += 1

    async def generate(client, i, text_with_index, semaphore, counter, start_time):
        async with semaphore:
            try:
                index, text = text_with_index
                content = await client.predict(text)
                response = (index, content, None)
            except Exception as e:
                print(f"{i}th request failed with exception: {e}")
                response = (index, None, 0, str(e))

            await counter.increment()
            if counter.value % config_logging_interval == 0:
                print(f"processed total {counter.value} requests in {time.time() - start_time:.2f} seconds.")
            return response

    async def batch_inference(texts_with_index):
        semaphore = asyncio.Semaphore(config_concurrency)
        counter = AsyncCounter()
        client = AsyncChatClient()
        start_time = time.time()

        tasks = [generate(client, i, text_with_index, semaphore, counter, start_time) for i, text_with_index in enumerate(texts_with_index)]
        responses = await asyncio.gather(*tasks)
        await client.close()

        return responses
        
    texts_with_index = [ (i, sent) for i, sent in enumerate(texts) ]

    responses = await batch_inference(texts_with_index)

    return responses

class SemanticTextChunker:
    """
    Splits text based on cosine distance of consecutive sentences, up to the specified max chunk size.

    :param max_chunk_size: the number of tokens for max length of a single chunk
    :param split_distance_percentile: consecutive sentences that have a cosine distance in above this percentile will be chunked
    :param min_sentences: The smallest grouping of sentences, if set to 1 splitting can be sensitive/noisy
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        split_distance_percentile: float = 0.95,
        min_sentences: int = 3,
    ):
        nltk.download("punkt")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME)
        # self.dbembeds = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
        self.max_chunk_size = max_chunk_size
        self.split_distance_percentile = split_distance_percentile
        self.min_sentences = min_sentences

    async def compute_embeddings_and_token_counts(self, sentences):
        processed_sentences = []
        texts = []

        for sentence in sentences:
            token_count = len(self.tokenizer.encode(sentence))
            if token_count <= self.max_chunk_size:
                processed_sentences.append(
                    {"sentence": sentence, "token_count": token_count}
                )
                texts.append(sentence)
            else:
                split_idx = len(sentence) // 2
                l, r = sentence[:split_idx], sentence[split_idx:]
                processed_sentences.append(
                    {"sentence": l, "token_count": len(self.tokenizer.encode(l))}
                )
                processed_sentences.append(
                    {"sentence": r, "token_count": len(self.tokenizer.encode(r))}
                )
                texts.extend([l, r])

        embds = await batch_serving_endpoint_embedding(texts)

        for emb, sent in zip(embds, processed_sentences):
            sent["embedding"] = emb[1]

        return processed_sentences

    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["embedding"]
            embedding_next = sentences[i + 1]["embedding"]

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]["distance_to_next"] = distance

        sentences[-1]["distance_to_next"] = 0

        return distances, sentences

    def segment_sentences(self, sentences, threshold, max_seq_len, n_sentences):
        groups = []
        current_group = []
        current_seq_len = 0
        sentence_count = 0

        for sentence in sentences:
            token_count = sentence["token_count"]
            distance_to_next = sentence.get("distance_to_next", 0)

            if not current_group:
                current_group.append(sentence)
                current_seq_len += token_count
                sentence_count = 1
            else:
                if current_seq_len + token_count <= max_seq_len and (
                    sentence_count % n_sentences != 0 or distance_to_next <= threshold
                ):
                    current_group.append(sentence)
                    current_seq_len += token_count
                    sentence_count += 1
                else:
                    groups.append(current_group)
                    current_group = [sentence]
                    current_seq_len = token_count
                    sentence_count = 1

        if current_group:
            groups.append(current_group)

        return groups

    async def chunk_parsed_content_async(
        self,
        doc_parser_output: Dict[str, str]
    ) -> ChunkerReturnValue:
        try: 
            parsed_content = doc_parser_output.get("parsed_content")
            sentences = nltk.sent_tokenize(parsed_content)

            if len(sentences) < self.min_sentences:
                return {
                    OUTPUT_COLUMN_NAME: sentences,
                    STATUS_COLUMN_NAME: "SUCCESS",
                }

            sentences = await self.compute_embeddings_and_token_counts(sentences)
            distances, sentences = self.calculate_cosine_distances(sentences)

            split_distance_threshold = np.percentile(
                distances, self.split_distance_percentile
            )
            sentence_groups = self.segment_sentences(
                sentences, split_distance_threshold, self.max_chunk_size, self.min_sentences
            )

            final_sentences = []

            for group in sentence_groups:
                g_text = " ".join([g["sentence"] for g in group])
                final_sentences.append(g_text)

            return {
                OUTPUT_COLUMN_NAME: final_sentences,
                STATUS_COLUMN_NAME: "SUCCESS",
            }
        except Exception as e:
            warnings.warn(f"Exception {e} has been thrown during parsing")
            return {
                OUTPUT_COLUMN_NAME: [],
                STATUS_COLUMN_NAME: f"ERROR: {e}",
            } 

    def chunk_parsed_content(
        self,
        doc_parsed_contents: str,
    ) -> ChunkerReturnValue:
        nltk.download("punkt")
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.chunk_parsed_content_async(doc_parsed_contents))
        except RuntimeError:  # No event loop in place
            return asyncio.run(self.chunk_parsed_content_async(doc_parsed_contents))

    def get_chunker_udf(self):
        chunker_udf = udf(
            self.chunk_parsed_content,
            returnType=StructType(
                [
                    StructField(OUTPUT_COLUMN_NAME, ArrayType(StringType()), nullable=True),
                    StructField(STATUS_COLUMN_NAME, StringType(), nullable=True),
                ]
            ),
        )
        return chunker_udf
