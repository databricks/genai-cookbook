from typing import Any, Dict, Tuple, Type
import yaml
from pydantic import BaseModel


def serializable_config_to_yaml(obj: BaseModel) -> str:
    data = obj.model_dump()
    return yaml.dump(data)


# The way serialization works:
# The goal of serialization is to save the class name (e.g., util.xx.xx.configClassName) with the dumped YAML.
# This allows ANY config to be dynamically loaded from a YAML without knowing about the configClassName before OR having it imported in your python env.
# This is necessary for MultiAgent.`agents` and FunctionCallingAgent.`tools` since they can have multiple types of agent or tool configs in them -- when the config is loaded in the serving or local env, we don't know what these configClassName will be ahead of time & we want to avoid importing them all in the python env.
# How it works:
# the ONLY way to dump a class is to call model_dump() on it, which will return a dict with the _CLASS_PATH_KEY key containing the class path e.g., util.xx.xx.configClassName
# all other dumping methods (yaml, etc) call model_dump() since it is a Pydantic method
# the ONLY way to load a serialized class is to call load_obj_from_yaml with the YAML string
# load_obj_from_yaml will parse the YAML string and get the class path key
# it will then use that class path key to dynamically load the class from the python path
# it will then call that class's _load_class_from_dict method with the remaining data to let it do anything custom e.g,. load the tools or the agents
# if you haven't overridden _load_class_from_dict, it will call the default implementation of this method from SerializableModel
# otherwise, it will call your overridden _load_class_from_dict method

# How to use:
# Inherit your config class from SerializableModel
# If you don't have any SerializableModel fields, you can just call load_obj_from_yaml directly on your class's dumped YAML string, nothing else required
# If you have SerializableModel fields, you need to
# 1. Override the _load_class_from_dict method to handle the deserialization of those sub-configs
# 2. Override the model_dump method to call the model_dump of each of those sub-configs properly
#
# Examples
# 1. No sub-configs: GenieAgentConfig, UCTool
# 2. Has sub-configs: FunctionCallingAgentConfig (in `tools`), MultiAgentConfig (in `agents`)
#  load_obj_from_yaml --> the only way a class is loaded, will get the class path key

# TODO: add tests.  this was tested manually in a notebook verifying that all classes worked.


_CLASS_PATH_KEY = "class_path"


class SerializableConfig(BaseModel):
    def to_yaml(self) -> str:
        return serializable_config_to_yaml(self)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped[_CLASS_PATH_KEY] = f"{self.__module__}.{self.__class__.__name__}"
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        return class_object(**data)

    def pretty_print(self):
        print(json.dumps(self.model_dump(), indent=2))


def serializable_config_to_yaml_file(obj: BaseModel, yaml_file_path: str) -> None:
    with open(yaml_file_path, "w") as handle:
        handle.write(serializable_config_to_yaml(obj))


# Helper method used by SerializableModel's with fields containing SerializableModels
def _load_class_from_dict(data: Dict[str, Any]) -> Tuple[Type, Dict[str, Any]]:
    """Dynamically load a class from data containing a class path.

    Args:
        data: Dictionary containing _CLASS_PATH_KEY and other data

    Returns:
        Tuple[Type, Dict[str, Any]]: The class object and the remaining data
    """
    class_path = data.pop(_CLASS_PATH_KEY)

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name), data


def load_serializable_config_from_yaml(yaml_str: str) -> SerializableConfig:
    data = yaml.safe_load(yaml_str)
    class_obj, remaining_data = _load_class_from_dict(data)
    return class_obj._load_class_from_dict(class_obj, remaining_data)


def load_serializable_config_from_yaml_file(yaml_file_path: str) -> SerializableConfig:
    with open(yaml_file_path, "r") as file:
        return load_serializable_config_from_yaml(file.read())
