from databricks.sdk import WorkspaceClient
from databricks import agents

def get_inference_tables(uc_model_fqn):
    w = WorkspaceClient()

    deployment = agents.get_deployments(uc_model_fqn)
    if len(deployment) == 0:
        raise ValueError(f"No deployments found for model {uc_model_fqn}")
    endpoint = w.serving_endpoints.get(deployment[0].endpoint_name)


    try:
        endpoint_config = endpoint.config.auto_capture_config
    except AttributeError as e:
        endpoint_config = endpoint.pending_config.auto_capture_config

    inference_table_name = endpoint_config.state.payload_table.name
    inference_table_catalog = endpoint_config.catalog_name
    inference_table_schema = endpoint_config.schema_name

    # Cleanly formatted tables
    assessment_log_table_name = f"{inference_table_name}_assessment_logs"
    request_log_table_name = f"{inference_table_name}_request_logs"

    return {
        'uc_catalog_name': inference_table_catalog,
        'uc_schema_name': inference_table_schema,
        'table_names': {
            'raw_payload_logs': inference_table_name,
            'assessment_logs': assessment_log_table_name,
            'request_logs': request_log_table_name,
        }
        
    }
