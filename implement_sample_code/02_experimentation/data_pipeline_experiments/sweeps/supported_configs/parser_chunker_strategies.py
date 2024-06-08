# Databricks notebook source
# MAGIC %md
# MAGIC ## Parsing strategies
# MAGIC
# MAGIC The below configurations can be swapped into the `file_format` and `parser` params inside the `pipeline_config` parameter in `00_config` <br/><br/>
# MAGIC ```
# MAGIC CONFIG_TO_RUN = "your_short_name" ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC configurations = {
# MAGIC     "your_short_name": { ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC         ...
# MAGIC
# MAGIC         "pipeline_config": {
# MAGIC             # REPLACE THE FILE FORMAT HERE
# MAGIC             "file_format": "pdf",
# MAGIC             # REPLACE THE PARSER HERE
# MAGIC             "parser": {"name": "pypdf", "config": {}},
# MAGIC
# MAGIC             ...
# MAGIC         },
# MAGIC     },
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### PDF files

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyPdf
# MAGIC
# MAGIC Parse a PDF with `pypdf` library.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "pdf",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "pypdf", "config": {}},
    # "chunker": {}
}


# COMMAND ----------

# MAGIC %md
# MAGIC #### PyMuPdf
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "pdf",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "pymupdf", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unstructured.IO
# MAGIC
# MAGIC Parse a PDF with `unstructured` library from unstructured.io.
# MAGIC
# MAGIC TODO: Add in the configuration options here.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "pdf",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "unstructuredPDF", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyMuPdfMarkdown
# MAGIC
# MAGIC Parse a PDF with `pymupdf` library, converting the output to Markdown.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "pdf",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "pymupdf_markdown", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Azure Doc Intelligence 
# MAGIC
# MAGIC TODO: Add the file types that are supported
# MAGIC
# MAGIC TODO: Add instructions for configuring the secrets, ideally move to the `config` param

# COMMAND ----------


pipeline_config = {
    # File format of the source documents
    "file_format": "pdf",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "azure_doc_intelligence", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## HTML

# COMMAND ----------

# MAGIC %md
# MAGIC #### HTML to Markdown 
# MAGIC
# MAGIC Use `markdownify` to parse an HTML file

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "html",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "html_to_markdown", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## DocX

# COMMAND ----------

# MAGIC %md
# MAGIC #### PyPandocDocx
# MAGIC
# MAGIC Parse a DocX file with Pandoc parser using the `pypandoc` library

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "docx",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "pypandocDocX", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unstructured.IO
# MAGIC
# MAGIC Parse a DocX with `unstructured` library from unstructured.io.
# MAGIC
# MAGIC TODO: Add in the configuration options here.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "docx",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "unstructuredDocX", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## PPTX

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #### Unstructured.IO
# MAGIC
# MAGIC Parse a PPTX with `unstructured` library from unstructured.io.
# MAGIC
# MAGIC TODO: Add in the configuration options here.

# COMMAND ----------

pipeline_config = {
    # File format of the source documents
    "file_format": "pptx",
    # Parser to use (must be present in `parser_library` Notebook)
    "parser": {"name": "unstructuredPPTX", "config": {}},
    # "chunker": {}
}

# COMMAND ----------

# MAGIC %md ## Chunking strategies
# MAGIC The below configurations can be swapped into the `chunker` param inside the `pipeline_config` parameter in `00_config` <br/><br/>```
# MAGIC CONFIG_TO_RUN = "your_short_name" ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC configurations = {
# MAGIC     "your_short_name": { ## REPLACE WITH A SHORT NAME TO IDENTIFY YOUR CONFIG
# MAGIC         ...
# MAGIC
# MAGIC         "pipeline_config": {
# MAGIC             ...
# MAGIC
# MAGIC             # REPLACE THE CHUNKER HERE
# MAGIC             "chunker": {
# MAGIC                 "name": "langchain_recursive_char",
# MAGIC                 "config": {
# MAGIC                     "chunk_size_tokens": 1500,
# MAGIC                     "chunk_overlap_tokens": 250,
# MAGIC                 },
# MAGIC             },
# MAGIC         },
# MAGIC     },
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #### Langchain Recursive Character Split
# MAGIC
# MAGIC Uses the embedding model's tokenizer to split the document into chunks.  The default implementation works for tiktoken and sentence_transformer based embedding models configured in `embedding_model_config`
# MAGIC
# MAGIC Per LangChain's docs: This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
# MAGIC
# MAGIC Configuration parameters:
# MAGIC - `chunk_size_tokens`: Number of tokens to include in each chunk
# MAGIC - `chunk_overlap_tokens`: Number of tokens to overlap between chunks e.g., the last `chunk_overlap_tokens` tokens of chunk N are the same as the first `chunk_overlap_tokens` tokens of chunk N+1
# MAGIC
# MAGIC Docs: https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
# MAGIC
# MAGIC **IMPORTANT: You need to ensure that `chunk_size_tokens` + `chunk_overlap_tokens` is LESS THAN your embedding model's context window.**

# COMMAND ----------

pipeline_config = {
    #"file_format": "...",
    #"parser": {}
    # Chunker to use (must be present in `chunker_library` Notebook)
    "chunker": {
        "name": "langchain_recursive_char",
        "config": {
            "chunk_size_tokens": 475,
            "chunk_overlap_tokens": 50,
        },
    },
}


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Langchain Markdown Header Splitter
# MAGIC
# MAGIC Uses Markdown headers to split the parsed document into chunks
# MAGIC
# MAGIC Configuration parameters:
# MAGIC - `include_headers_in_chunks`: if `True`, the markdown headers are included in each chunk.  If `False`, they are not.
# MAGIC
# MAGIC Docs: https://python.langchain.com/v0.2/docs/how_to/markdown_header_metadata_splitter/
# MAGIC
# MAGIC **IMPORTANT: This chunker does not check if the produced chunks will fit within you embedding model's context window.**

# COMMAND ----------

pipeline_config = {
    #"file_format": "...",
    #"parser": {}
    # Chunker to use (must be present in `chunker_library` Notebook)
    "chunker": {
        "name": "langchain_markdown_headers",
        "config": {
            # Include the markdown headers in each chunk?
            "include_headers_in_chunks": True,
        },
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Semantic chunk splitter
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
# MAGIC By default, this chunker uses the `databricks-gte-large` Embedding model, but you can replace this with any other Model Serving endpoint supporting the `/llm/v1/embeddings` signature such as an OpenAI External Model.  If you change the embedding model, make sure to update the tokenizer to match.  Modify this inside the `chunker_library` Notebook

# COMMAND ----------

pipeline_config = {
    #"file_format": "...",
    #"parser": {}
    # Chunker to use (must be present in `chunker_library` Notebook)
    "chunker": {
        "name": "semantic",
        "config": {
            # Include the markdown headers in each chunk?
            "max_chunk_size": 500,
            "split_distance_percentile": .95,
            "min_sentences": 3
        },
    },
}



def chunk_parsed_content_semantic(text: str, max_chunk_size: int = 512, split_distance_percentile: float = 0.95, min_sentences: int = 3) -> ChunkerReturnValue:
    """
    Splits text based on cosine distance of consecutive sentences, up to the specified max chunk size.

    :param max_chunk_size: the number of tokens for max length of a single chunk
    :param split_distance_percentile: consecutive sentences that have a cosine distance in above this percentile will be chunked
    :param min_sentences: The smallest grouping of sentences, if set to 1 splitting can be sensitive/noisy
    """
    chunker = SemanticTextChunker(max_chunk_size, split_distance_percentile, min_sentences)
    return chunker.chunk_parsed_content(text)
    
