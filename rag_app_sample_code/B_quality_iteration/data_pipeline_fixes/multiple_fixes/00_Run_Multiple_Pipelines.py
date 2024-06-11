# Databricks notebook source
# MAGIC %pip install -U -qqqq install pyyaml

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %run ./shared_utilities

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Parsing/Chunking strategy configuration
# MAGIC 1. Modify `baseline_strategy` to provide default values across all strategies.  
# MAGIC 2. Update `strategies_to_try` to add additional strategies.  These dictionaries will be merged such that all keys present in `strategies_to_try` overwrite the keys in `baseline_strategy`.  
# MAGIC
# MAGIC For example, if you have:
# MAGIC
# MAGIC > `baseline_strategy = {'a': {'x': 1, 'y': 2}}`
# MAGIC
# MAGIC > `strategies_to_try = {'a': {'x': 4}}`
# MAGIC
# MAGIC The resulting strategy will be:
# MAGIC > `strategy = {'a': {'x': 4, 'y': 2}}`
# MAGIC
# MAGIC You can dynamically generate `strategies_to_try` if you want to try multiple combinations e.g., one parser with 5 different chunking strategies.  See the bottom of the this notebook for an example.

# COMMAND ----------

baseline_strategy = {
    # Short name to identify this strategy by in the evaluation sweep
    "strategy_short_name": "baseline",
    # Initial configuration that matches the POC settings
    # Vector Search index configuration
    "vectorsearch_config": {
        # Pipeline execution mode.
        # TRIGGERED: If the pipeline uses the triggered execution mode, the system stops processing after successfully refreshing the source table in the pipeline once, ensuring the table is updated based on the data available when the update started.
        # CONTINUOUS: If the pipeline uses continuous execution, the pipeline processes new data as it arrives in the source table to keep vector index fresh.
        "pipeline_type": "TRIGGERED",
    },
    # Embedding model to use
    # Tested configurations are available in the `supported_configs/embedding_models` Notebook
    "embedding_config": {
        # Model Serving endpoint name
        "embedding_endpoint_name": "databricks-gte-large-en",
        "embedding_tokenizer": {
            # Name of the embedding model that the tokenizer recognizes
            "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
            # Name of the tokenizer, either `hugging_face` or `tiktoken`
            "tokenizer_source": "hugging_face",
        },
    },
    # Parsing and chunking configuration
    # Databricks provides several default implementations for parsing and chunking documents.  See supported configurations in the `supported_configs/parser_chunker_strategies` Notebook, which are repeated below for ease of use.
    # You can only enable a single `file_format` and `parser` at once
    # You can also add a custom parser/chunker implementation by following the instructions in README.md
    "pipeline_config": {
        # File format of the source documents
        "file_format": "pdf",
        # Parser to use (must be present in `parser_library` Notebook)
        "parser": {"name": "pypdf", "config": {}},
        # "parser": {"name": "pymupdf", "config": {}},
        # "parser": {"name": "pymupdf_markdown", "config": {}},
        # "parser": {
        #     "name": "unstructuredPDF",
        #     "config": {
        #         "strategy": "hi_res",  # optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #         "hi_res_model_name": "yolox",  # optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #         "use_premium_features": False,  # optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #         "api_key": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #         "api_url": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #     },
        # },
        ## DocX
        # File format of the source documents
        # "file_format": "docx",
        # Parser to use (must be present in `parser_library` Notebook)
        # "parser": {"name": "pypandocDocX", "config": {}},
        # "parser": {
        #        "name": "unstructuredDocX",
        #        "config": {
        #            "strategy" : "hi_res",           #optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #            "hi_res_model_name": "yolox",  #optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #            "use_premium_features": False,  #optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #            "api_key": "", #dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #            "api_url": "", #dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #        },
        # },
        ## PPTX
        # File format of the source documents
        # "file_format": "pptx",
        # Parser to use (must be present in `parser_library` Notebook)
        # "parser": {
        #        "name": "unstructuredPPTX",
        #        "config": {
        #            "strategy" : "hi_res",           #optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
        #            "hi_res_model_name": "yolox",  #optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
        #            "use_premium_features": False,  #optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
        #            "api_key": "", # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
        #            "api_url": "", # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        #       },
        # },
        ## HTML
        # "file_format": "html",
        # "parser": {"name": "html_to_markdown", "config": {}},
        # Chunker to use (must be present in `chunker_library` Notebook).  See supported configurations in the `supported_configs/parser_chunker_strategies` Notebook, which are repeated below for ease of use.
        ## JSON
        # "file_format": "json",
        # "parser": {
        #     "name": "json",
        #     "config": {
        #         # The key of the JSON file that contains the content that should be chunked
        #         # All other keys will be passed through
        #         "content_key": "html_content"
        #     },
        # },
        "chunker": {
            ## Split on number of tokens
            "name": "langchain_recursive_char",
            "config": {
                "chunk_size_tokens": 1500,
                "chunk_overlap_tokens": 250,
            },
            ## Split on Markdown headers
            # "name": "langchain_markdown_headers",
            # "config": {
            #     # Include the markdown headers in each chunk?
            #     "include_headers_in_chunks": True,
            # },
            ## Semantic chunk splitter
            # "name": "semantic",
            # "config": {
            #     # Include the markdown headers in each chunk?
            #     "max_chunk_size": 500,
            #     "split_distance_percentile": .95,
            #     "min_sentences": 3
            # },
            "output_table": {
                # The parser function returns a Dict[str, str].  If true, all keys in this dictionary other than 'parsed_content' will be included in the chunk table.  Use this if you extract metadata in your parser that you want to use as a filter in your Vector Search index.
                "include_parser_metadata_as_columns": False,
                # If true, the 'parsed_content' in the Dict[str, str] returned by the parser will be included in the chunk table.  `include_parser_metadata_in_chunk_table` must be True or this option is ignored.
                "include_parent_doc_content_as_column": False,
            },
        },
    },
}

# COMMAND ----------

# Array of strategies to try
strategies_to_try = [
    {
        # The `strategy_short_name` key is required
        # This should be a short name to identify this strategy by in the evaluation sweep
        "strategy_short_name": "example_1",
        # You can include any of the top-level keys in `baseline_strategy`
        "pipeline_config": {
            "file_format": "pdf",
            "parser": {"name": "pymupdf", "config": {}},
            "chunker": {
                # "name": "langchain_recursive_char",
                "config": {"chunk_size_tokens": 1500 },
            },
        },
    },
    {
        # The `strategy_short_name` key is required
        # This should be a short name to identify this strategy by in the evaluation sweep
        "strategy_short_name": "example_2",
        # You can include any of the top-level keys in `baseline_strategy`
        "embedding_config": {
            # Model Serving endpoint name
            "embedding_endpoint_name": "databricks-gte-large-en",
            "embedding_tokenizer": {
                # Name of the embedding model that the tokenizer recognizes
                "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
                # Name of the tokenizer, either `hugging_face` or `tiktoken`
                "tokenizer_source": "hugging_face",
            },
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the system packages required by parsers
# MAGIC
# MAGIC This will install the needed apt-get packages to run various parsers.  In the single data pipeline notebooks, this step is run during parsing/chunking, but since this pipeline runs multiple strategies at once, installing these packages within each strategy will cause a conflict.

# COMMAND ----------

install_apt_get_packages(["poppler-utils", "tesseract-ocr"])
install_apt_get_packages(['pandoc'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare and run the strategies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare the strategies to run

# COMMAND ----------

# This step "packs" each strategy into a JSON string that is passed to each Notebook as a parameter.
packed_strategies = []
for strategy in strategies_to_try:
    packed_strategies.append({
        "config": strategy,
        "packed_json": get_strategy_packed_json_string(baseline_strategy=baseline_strategy, strategy_to_try=strategy)
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute the strategies in parallel

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import json

NUMBER_THREADS = 5

# If True, the tables & streaming checkpoints will be deleted before starting.
# Use True if you are actively changing your parsing/chunking code and need to re-run for already processed files.
# If you re-ruse a `strategy_short_name`, you will need to set to True in order to reset, otherwise you will experience an error with Spark streaming
RESET_TABLES_BEFORE_RUNNING = True

def write_to_yaml(json_data, file_name):
    yaml_str = yaml.dump(json_data)
    yaml_file_path = f"output_chain_configs/{file_name}.yaml"

    with open(yaml_file_path, "w") as file:
        yaml.dump(json_data, file)

# Define the function to run the notebook
def run_notebook(notebook_path, timeout_seconds=0, args={}):
    result = dbutils.notebook.run(notebook_path, timeout_seconds, args)
    return result

def run_main_notebooks(packed_strategy_as_json):
    validate = run_notebook(notebook_path="00_validate_config", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
    if validate:
        load = run_notebook(notebook_path="01_load_files", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
        if load:
            parse = run_notebook(notebook_path="02_parse_docs", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
            if parse:
                chunk = run_notebook(notebook_path="03_chunk_docs", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
                if chunk:
                    index = run_notebook(notebook_path="04_vector_index", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
                    if index:
                        return True
    else:
        return False

def run_single_strategy(strategy):
    # print(strategy)
    packed_strategy_as_json = strategy['packed_json']
    print("----Start Run----")
    print("Strategy: " + strategy['config']['strategy_short_name'])

    if RESET_TABLES_BEFORE_RUNNING:
        reset = run_notebook(notebook_path="reset_tables_and_checkpoints", timeout_seconds=0, args={"strategy_to_run": packed_strategy_as_json})
        if reset:
            outcome = run_main_notebooks(packed_strategy_as_json)
        else:
            outcome = False
    else:
        outcome = run_main_notebooks(packed_strategy_as_json)

    if outcome is True:
        strategy['status'] = "success"
    else: 
        strategy['status'] = "failure"
       
    return strategy


# Use ThreadPoolExecutor to run notebooks in parallel
with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
    # Create a future for each notebook run
    futures = [executor.submit(run_single_strategy, strategy) for strategy in packed_strategies]
    
    # Process the results as they complete
    results = []
    for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['status'] == 'success':
                # print(result)
                print(f"Strategy `{result['config']['strategy_short_name']}` SUCCESS")
                # write_to_yaml(json_data=result['chain_config'], file_name=result['config']['strategy_short_name'])
            else:
                # print(result)
                print(f"Strategy `{result['config']['strategy_short_name']}` FAILED")
            print("----")

# Contains information about all runs
# print(results)

# Print out the resulting names of the MLflow Runs for each pipeline to copy / paste to the evaluation notebook
for item in strategies_to_try:
    print('"data_pipeline_'+item['strategy_short_name']+'",')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Sweeping multiple combinations of strategies
# MAGIC
# MAGIC **Important: Each Vector Search Endpoint has a quota of 20 indexes.  Make sure you have sufficient capacity in your Endpoint before running the sweep.  If you need more indexes per endpoint, please contact your Databricks represenative for additional quota.**
# MAGIC
# MAGIC Below is an example of sweeping across embedding models, parsers, and chunkers.

# COMMAND ----------

embedding_configs = {
    "gte_large": {
        "embedding_config": {
            # Model Serving endpoint name
            "embedding_endpoint_name": "databricks-gte-large-en",
            "embedding_tokenizer": {
                # Name of the embedding model that the tokenizer recognizes
                "tokenizer_model_name": "Alibaba-NLP/gte-large-en-v1.5",
                # Name of the tokenizer, either `hugging_face` or `tiktoken`
                "tokenizer_source": "hugging_face",
            },
        }
    },
    # "openai_small": {
    #     "embedding_config": {
    #         # Model Serving endpoint name
    #         "embedding_endpoint_name": "text-embedding-small",  # REPLACE WITH YOUR EXTERNAL MODEL ENDPOINT NAME
    #         "embedding_tokenizer": {
    #             # Name of the embedding model that the tokenizer recognizes
    #             "tokenizer_model_name": "text-embedding-small",
    #             # Name of the tokenizer, either `hugging_face` or `tiktoken`
    #             "tokenizer_source": "tiktoken",
    #         },
    #     }
    # },
}


parsing_strategies = {
    "pdf_pypdf": {
        "pipeline_config": {
            "file_format": "pdf",
            "parser": {"name": "pypdf", "config": {}},
        }
    },
    "parser": {
        "name": "unstructuredPDF",
        "config": {
            "strategy": "hi_res",  # optional; Strategy Options: "hi_res"[Default], "ocr_only", "fast"
            "hi_res_model_name": "yolox",  # optional; hi_res model name. Options  "yolox"[Default], "yolox_quantized", "detectron2_onnx"
            "use_premium_features": False,  # optional; allows user to toggle/on premium features on a per call basis. Options: True, False [Default] .
            "api_key": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_key"), #optional; needed for premium features
            "api_url": "",  # dbutils.secrets.get(scope="your_scope", key="unstructured_io_api_url"),  #optional; needed for premium features
        },
    },
    "pdf_pymupdf": {
        "pipeline_config": {
            "file_format": "pdf",
            "parser": {"name": "pymupdf", "config": {}},
        }
    },
    "pdf_pymupdf_markdown": {
        "pipeline_config": {
            "file_format": "pdf",
            "parser": {"name": "pymupdf_markdown", "config": {}},
        }
    },
    
}

chunking_strategies = {
    "langchain_recursive_char_512_256": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 512,
                    "chunk_overlap_tokens": 256,
                },
            },
        }
    },
    "langchain_recursive_char_1024_256": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 1024,
                    "chunk_overlap_tokens": 256,
                },
            },
        }
    },
    "langchain_recursive_char_2048_256": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 2048,
                    "chunk_overlap_tokens": 256,
                },
            },
        }
    },
    "langchain_recursive_char_2048_512": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 2048,
                    "chunk_overlap_tokens": 512,
                },
            },
        }
    },
    "langchain_recursive_char_4096_256": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 4096,
                    "chunk_overlap_tokens": 256,
                },
            },
        }
    },
    "langchain_recursive_char_4096_512": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 4096,
                    "chunk_overlap_tokens": 512,
                },
            },
        }
    },
    "langchain_recursive_char_8192_0": {
        "pipeline_config": {
            "chunker": {
                ## Split on number of tokens
                "name": "langchain_recursive_char",
                "config": {
                    "chunk_size_tokens": 8192,
                    "chunk_overlap_tokens": 0,
                },
            },
        }
    },
    # "langchain_markdown_headers_w_headers": {
    #     "pipeline_config": {
    #         "chunker": {
    #             ## Split on Markdown headers
    #             "name": "langchain_markdown_headers",
    #             "config": {
    #                 # Include the markdown headers in each chunk?
    #                 "include_headers_in_chunks": True,
    #             },
    #         },
    #     }
    # },
    # "langchain_markdown_headers_wout_headers": {
    #     "pipeline_config": {
    #         "chunker": {
    #             ## Split on Markdown headers
    #             "name": "langchain_markdown_headers",
    #             "config": {
    #                 # Include the markdown headers in each chunk?
    #                 "include_headers_in_chunks": False,
    #             },
    #         },
    #     }
    # },
    # "semantic_chunks_512": {
    #     "pipeline_config": {
    #         "chunker": {
    #             ## Semantic chunk splitter
    #             "name": "semantic",
    #             "config": {
    #                 # Include the markdown headers in each chunk?
    #                 "max_chunk_size": 512,
    #                 "split_distance_percentile": 0.95,
    #                 "min_sentences": 3,
    #             },
    #         },
    #     }
    # },
    # "semantic_chunks_1024": {
    #     "pipeline_config": {
    #         "chunker": {
    #             ## Semantic chunk splitter
    #             "name": "semantic",
    #             "config": {
    #                 # Include the markdown headers in each chunk?
    #                 "max_chunk_size": 1024,
    #                 "split_distance_percentile": 0.95,
    #                 "min_sentences": 3,
    #             },
    #         },
    #     }
    # },
    # # "semantic_chunks_2048": {
    # #     "pipeline_config": {
    # #         "chunker": {
    # #             ## Semantic chunk splitter
    # #             "name": "semantic",
    # #             "config": {
    # #                 # Include the markdown headers in each chunk?
    # #                 "max_chunk_size": 2048,
    # #                 "split_distance_percentile": 0.95,
    # #                 "min_sentences": 3,
    # #             },
    # #         },
    # #     }
    # # },
    # # "semantic_chunks_4096": {
    # #     "pipeline_config": {
    # #         "chunker": {
    # #             ## Semantic chunk splitter
    # #             "name": "semantic",
    # #             "config": {
    # #                 # Include the markdown headers in each chunk?
    # #                 "max_chunk_size": 4096,
    # #                 "split_distance_percentile": 0.95,
    # #                 "min_sentences": 3,
    # #             },
    # #         },
    # #     }
    # # },
}

# COMMAND ----------

# This cell is commented out to avoid overwriting the strategies you defined above in `strategies_to_try`.
# Uncomment to use.


strategies_to_try = []
for embedding_name, embedding in embedding_configs.items():
    for parsing_name, parser  in parsing_strategies.items():
            for chunking_name, chunker in chunking_strategies.items():
                temp = merge_dicts(parser, chunker)
                resulting_strategy = merge_dicts(temp, embedding)
                resulting_strategy['strategy_short_name'] = f'{parsing_name}_{chunking_name}_{embedding_name}'
                strategies_to_try.append(resulting_strategy)
                # print(resulting_strategy)

print(f"Created {len(strategies_to_try)} strategies.")
print("Example: ")
strategies_to_try

# This step "packs" each strategy into a JSON string that is passed to each Notebook as a parameter.
packed_strategies = []
for strategy in strategies_to_try:
    packed_strategies.append({
        "config": strategy,
        "packed_json": get_strategy_packed_json_string(baseline_strategy=baseline_strategy, strategy_to_try=strategy)
    })
