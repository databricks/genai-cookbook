from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
import mlflow
from .agent_storage_location_config import AgentStorageLocationConfig


def validate_storage_config(agent_storage_locations_config: AgentStorageLocationConfig):
  _validate_or_create_uc_catalog(agent_storage_locations_config)
  _validate_or_create_uc_schema(agent_storage_locations_config)
  _validate_or_create_mlflow_experiment(agent_storage_locations_config)

def _validate_or_create_uc_catalog(agent_storage_locations_config: AgentStorageLocationConfig):
  w = WorkspaceClient()

  # Create UC Catalog if it does not exist, otherwise, raise an exception
  try:
      _ = w.catalogs.get(agent_storage_locations_config.uc_catalog)
      print(f"PASS: UC catalog `{agent_storage_locations_config.uc_catalog}` exists")
  except NotFound as e:
      print(f"`{agent_storage_locations_config.uc_catalog}` does not exist, trying to create...")
      try:
          _ = w.catalogs.create(name=agent_storage_locations_config.uc_catalog)
      except PermissionDenied as e:
          print(
              f"FAIL: `{agent_storage_locations_config.uc_catalog}` does not exist, and no permissions to create.  Please provide an existing UC Catalog."
          )
          raise ValueError(f"Unity Catalog `{agent_storage_locations_config.uc_catalog}` does not exist.")

def _validate_or_create_uc_schema(agent_storage_locations_config: AgentStorageLocationConfig):
  w = WorkspaceClient()
  # Create UC Schema if it does not exist, otherwise, raise an exception
  try:
      _ = w.schemas.get(full_name=f"{agent_storage_locations_config.uc_catalog}.{agent_storage_locations_config.uc_schema}")
      print(f"PASS: UC schema `{agent_storage_locations_config.uc_catalog}.{agent_storage_locations_config.uc_schema}` exists")
  except NotFound as e:
      print(f"`{agent_storage_locations_config.uc_catalog}.{agent_storage_locations_config.uc_schema}` does not exist, trying to create...")
      try:
          _ = w.schemas.create(name=agent_storage_locations_config.uc_schema, catalog_name=agent_storage_locations_config.uc_catalog)
          print(f"PASS: UC schema `{agent_storage_locations_config.uc_catalog}.{agent_storage_locations_config.uc_schema}` created")
      except PermissionDenied as e:
          print(
              f"FAIL: `{agent_storage_locations_config.uc_catalog}.{agent_storage_locations_config.uc_schema}` does not exist, and no permissions to create.  Please provide an existing UC Schema."
          )
          raise ValueError(
              "Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist."
          )

def _validate_or_create_mlflow_experiment(agent_storage_locations_config):
    try:
        mlflow.set_experiment(agent_storage_locations_config.mlflow_experiment_directory)
        print(
            f"PASS: Using MLflow experiment directory `{agent_storage_locations_config.mlflow_experiment_directory}`."
        )
    except Exception as e:
        print(
            f"FAIL: `{agent_storage_locations_config.mlflow_experiment_directory}` is not a valid directory for an MLflow experiment.  An experiment name must be an absolute path within the Databricks workspace, e.g. '/Users/<some-username>/my-experiment'."
        )
        raise ValueError(f"MLflow experiment `{agent_storage_locations_config.mlflow_experiment_directory}` is not valid.")

# TODO: Add validation for the user having the correct permissions
# w.current_user.me().id

# # from databricks.sdk.types import SecurableType

# from databricks.sdk.service.catalog import SecurableType

# w.grants.get_effective(securable_type=SecurableType.CATALOG, principal=w.current_user.me().emails[0].value, full_name="adam_zhou")

