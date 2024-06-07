## Mode 2: Try across multiple strategies at once

To run the provided code for a set of documents in a specific volume, follow these steps.

### Configure the shared settings
__Configure your pipeline__ using the [00_config](./00_config.py) notebook which contains several python dictionaries.
1. Specify the constants:
    - `UC_CATALOG` and `UC_SCHEMA`: Unity Catalog Schema where the Delta Tables with the parsed/chunked documents are stored
    - `VECTOR_SEARCH_ENDPOINT`: Vector Search endpoint to host the resulting vector index.
    - `SOURCE_PATH`: Unity Catalog Volume with the source documents to process
    - (optional) `CHECKPOINTS_VOLUME_PATH`: Unity Catalog Volume to store Spark streaming checkpoints

### Configure data preparation strategies
1. Open [-01_Try_Multiple_Strategies](./-01_Try_Multiple_Strategies.py)
2. Follow the instructions in the notebook to configure the multiple strategies.  Each strategy is defined by the following:
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

        
    - __`pipeline_config`__: Defines the file parser, chunker and path to the sources field. Parsers and chunkers are defined in the [parser_library](./parser_library.py) and [chunker_library](./chunker_library.py) notebooks, respectively. For a complete list of options see the [`supporting_configs/parser_chunker_strategies`](./supporting_configs/parser_chunker_strategies) Notebook.  Different parsers or chunkers may require different configuration parameters, e.g.

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

### Run the pipeline

Run the notebook to execute the configured pipelines.  They will run in parallel.  

### Use the resulting vector index in a chain

Once finished, the YAML configurations for your chains are outputted to `output_chain_configs`.  Follow [link to chain readme] to evaluate these strategies.


## Appendix

See the README.md in the `../data_prep` folder for additional information on adding a new parsing or chunking strategy.