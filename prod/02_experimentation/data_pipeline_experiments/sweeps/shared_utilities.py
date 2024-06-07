# Databricks notebook source
# MAGIC %md
# MAGIC Utilities to help with strategy sweeps

# COMMAND ----------

import json
def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def get_strategy_packed_json_string(baseline_strategy, strategy_to_try):
    merged_strategy = merge_dicts(baseline_strategy, strategy_to_try)
    strategy_name = merged_strategy['strategy_short_name']

    # Names of the output Delta Tables tables & Vector Search index
    merged_strategy["destination_tables_config"] = {
        # Staging table with the raw files & metadata
        "raw_files_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{strategy_name}_raw_files_bronze",
        # Parsed documents
        "parsed_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{strategy_name}_parsed_docs_silver",
        # Chunked documents that are loaded into the Vector Index
        "chunked_docs_table_name": f"{UC_CATALOG}.{UC_SCHEMA}.{strategy_name}_chunked_docs_gold",
        # Destination Vector Index
        "vectorsearch_index_name": f"{UC_CATALOG}.{UC_SCHEMA}.{strategy_name}_chunked_docs_gold_index",
        # Streaming checkpoints, used to only process each file once
        "checkpoint_path": f"{CHECKPOINTS_VOLUME_PATH}/{strategy_name}/",
    }

    return json.dumps(merged_strategy)


def unpack_strategy(packed_strategy_json_string):
    strategy = json.loads(packed_strategy_json_string)
    vectorsearch_config = strategy["vectorsearch_config"]
    embedding_config = strategy["embedding_config"]
    pipeline_config = strategy["pipeline_config"]
    destination_tables_config = strategy["destination_tables_config"]

    mlflow_run_name = f"data_pipeline_{strategy['strategy_short_name']}"

    print(f"Using strategy: {strategy['strategy_short_name']}\n")
    print(f"Strategy settings: {json.dumps(strategy, indent=4)}\n")

    return (
        vectorsearch_config,
        embedding_config,
        pipeline_config,
        destination_tables_config,
        strategy,
        mlflow_run_name
    )

# COMMAND ----------

def load_strategy_from_widget(packed_strategy):

    if packed_strategy is not None and packed_strategy != "":
        return unpack_strategy(packed_strategy)

# COMMAND ----------

from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    if 'destination_tables_config' in flat_config:
        del flat_config['destination_tables_config']
    counter = 0
    for key, item in flat_config.items():
        if counter <19:
            sqls.append(f"""
            ALTER TABLE {table_fqn}
            SET TAGS ("{key.replace("/", "__")}" = "{item}")
            """)
        counter = counter + 1
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_data_prep_sweep")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)

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

# Helper to get an existing Mlflow run by name -- and if not exists - start it 

import mlflow
def get_or_start_mlflow_run(experiment_name, run_name):
    # Get the POC's data pipeline configuration
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"run_name = '{run_name}'", output_format="list")

    if len(runs) == 1:
        return mlflow.start_run(run_id=runs[0].info.run_id)
    elif len(runs) >1:
        raise ValueError("There are multiple runs named {run_name} in the experiment {experiment_name}.  Remove the additional runs or choose a different run name.")
    else:
        return mlflow.start_run(run_name=run_name)
