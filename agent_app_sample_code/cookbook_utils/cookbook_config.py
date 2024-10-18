from pydantic import BaseModel, Field, root_validator, computed_field, field_validator, FieldValidationInfo
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceAlreadyExists, ResourceDoesNotExist, NotFound, PermissionDenied

class AgentCookbookConfig(BaseModel):
    """
    Global configuration for an Agent.

    Attributes:
        uc_catalog_name (str):
          Required. Unity Catalog where the Agent's resources are stored.").
        uc_schema_name (str): 
          Required. Unity Catalog schema where the Agent's resources are stored..
        uc_asset_prefix (str): 
          Required if using default asset names. Prefix for the UC objects that will be created within the schema.  Typically a short name to identify the Agent e.g., "my_agent_app", used to generate the default names for `mlflow_experiment_name`, `evaluation_set_table`, `uc_model`.
        mlflow_experiment_name (str): 
          Optional.  The directory where the Agent's MLflow Experiment is stored. Defaults to '/Users/<curent-user-name>/{uc_asset_prefix}_mlflow_experiment'.
        uc_model (str):
          Optional.  UC location of the Unity Catalog model where versions of the Agent are registered.  Default is `{uc_asset_prefix}_model`
        evaluation_set_table (str): 
          Optional.  UC location of the Delta Table containing the Agent's evaluation set.  Default is `{uc_asset_prefix}_evaluation_set`
    """
    uc_catalog_name: str
    uc_schema_name: str 
    uc_asset_prefix: str = Field(None)
    mlflow_experiment_name: str = Field(None)
    uc_model: str = Field(None)
    evaluation_set_table: str = Field(None)

    def model_post_init(self, __context):
        if self.are_any_uc_asset_names_empty() and self.uc_asset_prefix is None:
            raise ValueError(
                "Must provide `uc_asset_prefix` since you did not provide a value for one or more of `mlflow_experiment_name`, `uc_model`, or `evaluation_set_table`.  `uc_asset_prefix` is used to compute the default values for these properties."
            )

        if self.uc_model is None:
            self.uc_model = self.get_uc_fqn(f"model")
        else:
            # if not a fully qualified UC path with catalog & schema, add the catalog & schema
            if self.uc_model.count(".") != 2:
                self.uc_model = self.get_uc_fqn_for_asset_name(
                    self.uc_model
                )

        if self.evaluation_set_table is None:
            self.evaluation_set_table = self.get_uc_fqn(f"evaluation_set")
        else:
            # if not a fully qualified UC path with catalog & schema, add the catalog & schema
            if self.evaluation_set_table.count(".") != 2:
                self.evaluation_set_table = self.get_uc_fqn_for_asset_name(
                    self.evaluation_set_table
                )

        if self.mlflow_experiment_name is None:
            print("No `mlflow_experiment_name` provided, attempting to infer from current user's home directory")
            w = WorkspaceClient()
            user_email = w.current_user.me().user_name
            user_home_directory = f"/Users/{user_email}"
            self.mlflow_experiment_name = f"{user_home_directory}/{self.uc_asset_prefix}_mlflow_experiment"

    def are_any_uc_asset_names_empty(self) -> bool:
        """
        Check if any of the Unity Catalog asset names are empty.

        Returns:
            bool: True if any of the asset names (`mlflow_experiment_name`, `uc_model`, or `evaluation_set_table`) are None, otherwise False.
        """
        if (
            self.mlflow_experiment_name is None
            or self.uc_model is None
            or self.evaluation_set_table is None
        ):
            return True
        else:
            return False
          
    def get_uc_fqn(self, asset_name: str) -> str:
        """
        Generate the fully qualified name (FQN) for a Unity Catalog asset.

        Args:
            asset_name (str): The name of the asset to generate the FQN for.

        Returns:
            str: The fully qualified name of the asset, with necessary escaping for special characters.
        """
        uc_fqn = f"{self.uc_catalog_name}.{self.uc_schema_name}.{self.uc_asset_prefix}_{asset_name}"

        return self.escape_uc_fqn(uc_fqn)

    def get_uc_fqn_for_asset_name(self, asset_name):
        uc_fqn = f"{self.uc_catalog_name}.{self.uc_schema_name}.{asset_name}"

        return self.escape_uc_fqn(uc_fqn)

    def escape_uc_fqn(self, uc_fqn):
        """
        Escape the fully qualified name (FQN) for a Unity Catalog asset if it contains special characters.

        Args:
            uc_fqn (str): The fully qualified name of the asset.

        Returns:
            str: The escaped fully qualified name if it contains special characters, otherwise the original FQN.
        """
        if "-" in uc_fqn:
            parts = uc_fqn.split(".")
            escaped_parts = [f"`{part}`" for part in parts]
            return ".".join(escaped_parts)
        else:
            return uc_fqn
    

    def dump_to_yaml(self, file_path: str):
        import yaml
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file)

    @classmethod
    def from_yaml_file(self, file_path: str):
        import yaml
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            # print(data)
        config = AgentCookbookConfig.parse_obj(data)
        # print(config.model_dump())
        return config
    
    def pretty_print(self):
        config_dump = self.model_dump()
        print(json.dumps(config_dump, indent=4))
      

    def validate_or_create_uc_catalog(self):
      w = WorkspaceClient()
      try:
          w.catalogs.get(self.uc_catalog_name)
      except NotFound:
          print(f"\n`{self.uc_catalog_name}` does not exist, trying to create...")
          w.catalogs.create(name=self.uc_catalog_name)

    def validate_or_create_uc_schema(self) -> bool:
      w = WorkspaceClient()
      try:
          _ = w.schemas.get(full_name=f"{self.uc_catalog_name}.{self.uc_schema_name}")
      except NotFound:
          print(f"\n`{self.uc_catalog_name}.{self.uc_schema_name}` does not exist, trying to create...")
          w.schemas.create(name=self.uc_schema_name, catalog_name=self.uc_catalog_name)

    def validate_or_create_mlflow_experiment(self):
        mlflow.set_experiment(self.mlflow_experiment_name)
