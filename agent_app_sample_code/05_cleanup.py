# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 🚨🚨🚨🚨 Delete all resources
# MAGIC
# MAGIC This notebook will print all resources created by these cookbook notebooks so you can remove them if desired.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install Python libraries

# COMMAND ----------

# Versions of Databricks code are not locked since Databricks ensures changes are backwards compatible.
%pip install -qqqq -U -r requirements.txt
# Restart to load the packages into the Python environment
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the Agent's configuration from other notebooks 
# MAGIC
# MAGIC We use this configuration to identify the resources.

# COMMAND ----------

from cookbook_utils.cookbook_config import AgentCookbookConfig
from datapipeline_utils.data_pipeline_config import UnstructuredDataPipelineSourceConfig, UnstructuredDataPipelineStorageConfig

# Load the shared configuration
cookbook_shared_config = AgentCookbookConfig.from_yaml_file('./configs/cookbook_config.yaml')

# Data pipeline configuration
datapipeline_source_config = UnstructuredDataPipelineSourceConfig.from_yaml_file('./configs/data_pipeline_source_config.yaml')
datapipeline_output_config = UnstructuredDataPipelineStorageConfig.from_yaml_file('./configs/data_pipeline_storage_config.yaml')

# COMMAND ----------

# DBTITLE 1,Shared imports
from mlflow.utils import databricks_utils as du
from cookbook_utils.shared import get_table_url
from IPython.display import display_markdown
import mlflow
from IPython.display import display_markdown

# Workspace URL
browser_url = f"https://{du.get_browser_hostname()}"

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Data pipeline
# MAGIC

# COMMAND ----------

output = f"""

## Data pipeline inputs

- Source UC volume: [{datapipeline_source_config.volume_path}]({get_table_url(datapipeline_source_config.volume_uc_fqn)})


## Data pipeline outputs

- Parsed docs table: [{datapipeline_output_config.parsed_docs_table}]({get_table_url(datapipeline_output_config.parsed_docs_table)})
- Chunked docs table: [{datapipeline_output_config.chunked_docs_table}]({get_table_url(datapipeline_output_config.chunked_docs_table)})
- Vector search index: [{datapipeline_output_config.vector_index}]({get_table_url(datapipeline_output_config.vector_index)})

*Note: The above output tables are from the last run of the data pipeline.  If you have run the notebook multiple times with different `tag` values, check the Unity Catalog schema [{datapipeline_output_config.uc_catalog_name}.{datapipeline_output_config.uc_schema_name}]({browser_url}/explore/data/{datapipeline_output_config.uc_catalog_name}/{datapipeline_output_config.uc_schema_name}) for other tables & indexes.*

## Compute resources
- Vector search endpoint: [{datapipeline_output_config.vector_search_endpoint}]({browser_url}/compute/vector-search/{datapipeline_output_config.vector_search_endpoint})
"""

display_markdown(output, raw=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Agent
# MAGIC
# MAGIC To remove the Model Serving endpoint, use the following code.  Deleting the endpoint from the UI will not remove all resources.<br/><br/>
# MAGIC
# MAGIC ```
# MAGIC from databricks import agents
# MAGIC
# MAGIC agents.delete_deployment(agent_storage_location_config.uc_model_fqn)
# MAGIC ```

# COMMAND ----------

from cookbook_utils.get_inference_tables import get_inference_tables
from databricks import agents

inference_tables = get_inference_tables(cookbook_shared_config.uc_model)

output = f"""

## Outputs

- Agent model: [{cookbook_shared_config.uc_model}]({get_table_url(cookbook_shared_config.uc_model).replace("explore/data/", "explore/data/models/")})
- MLflow experiment: [{cookbook_shared_config.mlflow_experiment_name}]({browser_url}/ml/experiments/{mlflow.get_experiment_by_name(cookbook_shared_config.mlflow_experiment_name).experiment_id})
- Evaluation set: [{cookbook_shared_config.evaluation_set_table}]({get_table_url(cookbook_shared_config.evaluation_set_table)})

## Inference table logs

- Raw logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['raw_payload_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['raw_payload_logs']}")})
- Request logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['request_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['request_logs']}")})
- Feedback logs: [{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['assessment_logs']}]({get_table_url(f"{inference_tables['uc_catalog_name']}.{inference_tables['uc_schema_name']}.{inference_tables['table_names']['assessment_logs']}")})


*Note: The above output tables are from the last run of the data pipeline.  If you have run the notebook multiple times with different `tag` values, check the Unity Catalog schema [{datapipeline_output_config.uc_catalog_name}.{datapipeline_output_config.uc_schema_name}]({browser_url}/explore/data/{datapipeline_output_config.uc_catalog_name}/{datapipeline_output_config.uc_schema_name}) for other tables & indexes.*

## Compute resources
- Model Serving endpoint: [{agents.get_deployments(cookbook_shared_config.uc_model)[0].endpoint_name}]({agents.get_deployments(cookbook_shared_config.uc_model)[0].endpoint_url
})

*Note: Deleting the endpoint will also remove the Review App.*

Use the following code to delete the endpoint:
```
from databricks import agents

agents.delete_deployment(cookbook_shared_config.uc_model)
```
"""

display_markdown(output, raw=True)
