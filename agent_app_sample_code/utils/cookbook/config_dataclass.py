from dataclasses import dataclass, asdict
import yaml
import json

@dataclass
class CookbookConfig:
    """
    A base class to add utility methods to a dataclass for handling YAML serialization and pretty printing.

    This class adds the following methods to the dataclass:
    - dump_to_yaml: Dumps the dataclass instance to a YAML file.
    - from_yaml_file: Loads a dataclass instance from a YAML file.
    - pretty_print: Prints the dataclass instance in a pretty JSON format.

    Returns:
        type: The dataclass with the added methods.
    """
    def dump_to_yaml(self, file_path: str):
        """
        Dump the configuration to a YAML file.

        Args:
            file_path (str): The path to the YAML file where the configuration will be saved.
        """
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file)

    @classmethod
    def from_yaml_file(cls, file_path: str):
        """
        Load the configuration from a YAML file.

        Args:
            file_path (str): The path to the YAML file from which the configuration will be loaded.

        Returns:
            UnstructuredDataPipelineStorageConfig: An instance of the configuration class populated with values from the YAML file.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            # print(data)
        return cls(**data)

    def pretty_print(self):
        config_dump = asdict(self)
        print(json.dumps(config_dump, indent=4))

 
