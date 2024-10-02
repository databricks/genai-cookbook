# **![Data pipeline](../images/5-hands-on/data_pipeline.png)** Implement data pipeline fixes

```{admonition} [Code Repository](https://github.com/databricks/genai-cookbook/tree/v0.2.0/agent_app_sample_code)
:class: tip
You can find all of the sample code referenced throughout this section [here](https://github.com/databricks/genai-cookbook/tree/v0.2.0/agent_app_sample_code).
```

#### Configuration settings deep dive

The various pre-implemented configuration options for the data pipeline are listed below.  Alternatively, you can implement a [custom parser/chunker](#implementing-a-custom-parser-chunker).

- __`vectorsearch_config`__: Specify the [vector search](https://docs.databricks.com/en/generative-ai/vector-search.html) endpoint (must be up and running) and the name of the index to be created. Additionally, define the [synchronisation](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-index-using-the-ui) type between the source table and the index (default is `TRIGGERED`).
  - __`embedding_config`__: Specify the embedding model to be used, along with the tokenizer. For a complete list of options see the [`supporting_configs/embedding_models`](./supporting_configs/embedding_models) Notebook.  The embedding model has to be deployed to a running [model serving endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search). Depending on chunking strategy, the tokenizer is also during splitting to make sure the chunks do not exceed the token limit of the embedding model.  Tokenizers are used here to count the number of tokens in the text chunks to ensure that they don't exceed the maximum context length of the selected embedding model. Tokenizers from HuggingFace or TikToken can be selected, e.g.

      ```Python
          "embedding_tokenizer": {
              "tokenizer_model_name": "BAAI/bge-large-en-v1.5",
              "tokenizer_source": "hugging_face",
          }
          ```

      or

      ```Python
      "embedding_tokenizer": {
              "tokenizer_model_name": "text-embedding-small",
              "tokenizer_source": "tiktoken",
          }
      ```

      
  - __`pipeline_config`__: Defines the file parser, chunker and path to the sources field. Parsers and chunkers are defined in the `parser_library` and `chunker_library` notebooks, respectively. These can be found in the [`single_fix`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/B_quality_iteration/data_pipeline_fixes/single_fix) and [`multiple_fixes`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/B_quality_iteration/data_pipeline_fixes/multiple_fixes) directories. For a complete list of options see the `supporting_configs/parser_chunker_strategies` Notebook, which is again available in both the single and multiple fix directories. Different parsers or chunkers may require different configuration parameters, e.g.

      ```Python
          "chunker": {
              "name": <chunker-name>,
              "config": {
                  "<param 1>": "...",
                  "<param 2>": "...",
                  ...
              }
          }
      ```

      where `<param x>` represent the potential parameters required for a specific chunker. Parsers can also be passed configuration values using the same format.


#### Implementing a custom parser/chunker
This project is structured to facilitate the addition of custom parsers or chunkers to the data preparation pipeline.

##### Add a new parser
Suppose you want to incorporate a new parser using the [PyMuPDF library](https://pypi.org/project/PyMuPDF/) to transform parsed text into Markdown format. Follow these steps:
1. Install the required dependencies by adding the following code to the `parser_library` notebook in the `single_fix` or `multiple_fix` directory:

    ```Python
    # Dependencies for PyMuPdf
    %pip install pymupdf pymupdf4llm
    ```

2. In the `parser_library` notebook in the `single_fix` or `multiple_fix` directory, add a new section for the `PyMuPdfMarkdown` parser and implement the parsing function: 

    ```Python
    import fitz
    import pymupdf4llm

    def parse_bytes_pymupdfmarkdown(
        raw_doc_contents_bytes: bytes,
    ) -> ParserReturnValue:
        try:
            pdf_doc = fitz.Document(stream=raw_doc_contents_bytes, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(pdf_doc)

            output = {
                "num_pages": str(pdf_doc.page_count),
                "parsed_content": md_text.strip(),
            }

            return {
                OUTPUT_FIELD_NAME: output,
                STATUS_FIELD_NAME: "SUCCESS",
            }
        except Exception as e:
            warnings.warn(f"Exception {e} has been thrown during parsing")
            return {
                OUTPUT_FIELD_NAME: {"num_pages": "", "parsed_content": ""},
                STATUS_FIELD_NAME: f"ERROR: {e}",
            }
    ```

    Ensure the output of the function complies with the `ParserReturnValue` class defined at the beginning of the notebook. This ensures compatibility with Spark UDFs. The `try`/`except` block prevents Spark from failing the entire parsing job due to errors in individual documents when applying the parser as a UDF in `02_parse_docs` notebook in the `single_fix` or `multiple_fix` directory. This notebook will check if parsing failed for any document, quarantine the corresponding rows and raise a warning.

3. Add your new parsing function to the `parser_factory` in the `parser_library` notebook in the `single_fix` or `multiple_fix` directory to make it configurable in the `pipeline_config` of the `00_config` notebook. 

4.  In the `02_parse_docs` notebook, parser functions are turned into Spark Python UDFs ([arrow-optimized](https://www.databricks.com/blog/arrow-optimized-python-udfs-apache-sparktm-35) for DBR >= 14.0) and applied to the dataframe containing the new binary PDF files. For testing and development, add a simple testing function to the [parser_library notebook](./parser_library.py) that loads the [test-document.pdf](./test_data/test-document.pdf) file and asserts successful parsing:

    ```python
    with open("./test_data/test-document.pdf", "rb") as file:
        file_bytes = file.read()
        test_result_pymupdfmarkdown = parse_bytes_pymupdfmarkdown(file_bytes)

    assert test_result_pymupdfmarkdown[STATUS_FIELD_NAME] == "SUCCESS"
    ```

##### Add a New Chunker
The process for adding a new chunker follows similar steps to those explained above for a new parser.

1. Add the required dependencies in the [chunker_library](./chunker_library.py) notebook.

2. Add a new section for your chunker and implement a function, e.g., `chunk_parsed_content_newchunkername`. The output of the new chunker function must be a Python dictionary that complies with the `ChunkerReturnValue` class defined at the beginning of the [chunker_library](./chunker_library.py) notebook. The function should accept at least a string of the parsed text to be chunked. If your chunker requires additional parameters, you can add them as function parameters.

3. Add your new chunker to the `chunker_factory` function defined in the [chunker_library](./chunker_library.py) notebook. If your function accepts additional parameters, use [functools' partial](https://docs.python.org/3/library/functools.html#functools.partial) to pre-configure them. This is necessary because UDFs only accept one input parameter, which will be the parsed text in our case. The `chunker_factory` enables you to configure different chunker methods in the [pipeline_config](./00_config.py) and returns a Spark Python UDF (optimized for DBR >= 14.0).

4. Add a simple testing section for your new chunking function. This section should chunk a predefined text provided as a string.



#### Performance Tuning
Spark utilizes partitions to parallelize processing. Data is divided into chunks of rows, and each partition is processed by a single core by default. However, when data is initially read by Apache Spark, it may not create partitions optimized for the desired computation, particularly for our UDFs performing parsing and chunking tasks. It's crucial to strike a balance between creating partitions that are small enough for efficient parallelization and not so small that the overhead of managing them outweighs the benefits.

You can adjust the number of partitions using `df.repartitions(<number of partitions>)`. When applying UDFs, aim for a multiple of the number of cores available on the worker nodes. For instance, in the [02_parse_docs](./02_parse_docs.py) notebook, you could include `df_raw_bronze = df_raw_bronze.repartition(2*sc.defaultParallelism)` to create twice as many partitions as the number of available worker cores. Typically, a multiple between 1 and 3 should yield satisfactory performance.

#### Running the pipeline manually

Alternatively, you can run each individual Notebook step-by-step:

1. __Load the raw files__ using the `01_load_files` notebook. This saves each document binary as one record in a bronze table (`raw_files_table_name`) defined in the `destination_tables_config`. Files are loaded incrementally, processing only new documents since the last run.

2. __Parse the documents__ with the `02_parse_docs` notebook. This notebook executes the `parser_library` notebook (*ensure to run this as the first cell to restart Python*), making different parsers and related utilities available. It then uses the specified parser in the `pipeline_config` to parse each document into plain text. 

> As an example, relevant metadata like the number of pages of the original PDF alongside the parsed text is captured. Successfully parsed documents are stored in a silver table (`parsed_docs_table_name`), while any unparsed documents are quarantined into a corresponding table.

3. __Chunk the parsed documents__ using the `03_chunk_docs` notebook. Similar to parsing, this notebook executes the `chunker_library` notebook (*again, run as the first cell*). It splits each parsed document into smaller chunks using the specified chunker from the `pipeline_config`. Each chunk is assigned a unique ID using an MD5 hash, necessary for synchronization with the vector search index. The final chunks are loaded into a gold table (`chunked_docs_table_name`).

4. __Create/Sync the vector search index__ with the `04_vector_index`. This notebook verifies the readiness of the specified vector search endpoint in the `vectorsearch_config`. If the configured index already exists, it initiates synchronization with the gold table; otherwise, it creates the index and triggers synchronization. This is expected to take some time if the Vector Search endpoint and index have not yet been created
