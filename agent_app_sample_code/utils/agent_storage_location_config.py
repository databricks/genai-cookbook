from pydantic import BaseModel, Field, root_validator, computed_field, field_validator, FieldValidationInfo
import os
import json

from databricks.sdk import WorkspaceClient

class AgentStorageLocationConfig(BaseModel):
    """
    Global configuration for an Agent.

    Attributes:
        uc_catalog (str): Unity Catalog where the Agent's resources are stored.").
        uc_schema (str): Unity Catalog schema where the Agent's resources are stored..
        uc_asset_prefix (str): Prefix for the UC assets created by these notebooks.  This is typically a short name to identify the Agent e.g., "my_agent_app".  Used  usage: '`{uc_catalog}`.`{uc_schema}`.`{uc_asset_prefix}`_example_table_name'.
        mlflow_experiment_directory (str): The directory where the Agent's MLflow Experiment is stored. Defaults to '/Users/<curent-user-name>/{uc_asset_prefix}_mlflow_experiment'.
    """
    uc_catalog: str
    uc_schema: str 
    uc_asset_prefix: str
    mlflow_experiment_directory: str = Field(None)

    def model_post_init(self, __context):
        if self.mlflow_experiment_directory is None:
            try:
              w = WorkspaceClient()

              user_email = w.current_user.me().user_name #spark.sql("SELECT current_user() as username").collect()[0].username 
            #   print(user_email)
              user_home_directory = f"/Users/{user_email}"
              
              self.mlflow_experiment_directory = f"{user_home_directory}/{self.uc_asset_prefix}_mlflow_experiment"
            except Exception as e:
            #   print(e)
              raise ValueError(f"Failed to identify the current user's working directory, which is used to initialize the default value for `mlflow_experiment_directory`.  Please explicitly specify `mlflow_experiment_directory` and retry.")

    def get_uc_fqn(self, asset_name:str) -> str:
        uc_fqn = f"{self.uc_catalog}.{self.uc_schema}.{self.uc_asset_prefix}_{asset_name}"

        # only escape the FQN if we need to per https://docs.databricks.com/en/sql/language-manual/sql-ref-names.html
        # TODO: Check for non-ascii chars which also need to be escaped
        
        if '-' in uc_fqn:
            parts = uc_fqn.split('.')
            escaped_parts = [f"`{part}`" for part in parts]
            return '.'.join(escaped_parts)
        else:
            return uc_fqn

    @computed_field
    def uc_model_fqn(self) -> str:
        return self.get_uc_fqn(f'model1')
    
    @computed_field
    def evaluation_set_fqn(self) -> str:
        return self.get_uc_fqn(f'evaluation_set')
    

    def dump_to_yaml(self, file_path: str):
        import yaml
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file)

    @staticmethod
    def from_yaml_file(file_path: str):
        import yaml
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            # print(data)
        config = AgentStorageLocationConfig.parse_obj(data)
        # print(config.model_dump())
        return config
    
    def pretty_print(self):
        config_dump = self.model_dump()
        print(json.dumps(config_dump, indent=4, sort_keys=True))
      
