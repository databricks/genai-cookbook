from typing import Optional, Any, Callable, get_type_hints, Type, Dict, Tuple
from pydantic import create_model, Field, BaseModel
import inspect
from unitycatalog.ai.core.utils.docstring_utils import parse_docstring
import yaml
import importlib
import mlflow
import json

_CLASS_PATH_KEY = "class_path"

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


class SerializableModel(BaseModel):
    def to_yaml(self) -> str:
        return obj_to_yaml(self)

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
    ) -> "SerializableModel":
        return class_object(**data)


def obj_to_yaml(obj: BaseModel) -> str:
    data = obj.model_dump()
    return yaml.dump(data)


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


def load_obj_from_yaml(yaml_str: str) -> SerializableModel:
    data = yaml.safe_load(yaml_str)
    class_obj, remaining_data = _load_class_from_dict(data)
    return class_obj._load_class_from_dict(class_obj, remaining_data)


def load_obj_from_yaml_file(yaml_file_path: str) -> SerializableModel:
    with open(yaml_file_path, "r") as file:
        return load_obj_from_yaml(file.read())


class Tool(SerializableModel):
    """Base class for all tools"""

    def __call__(self, **kwargs) -> Any:
        """Execute the tool with validated inputs"""
        raise NotImplementedError(
            "__call__ must be implemented by Tool subclasses. This method should execute "
            "the tool's functionality with the provided validated inputs and return the result."
        )

    name: str
    description: str

    def get_json_schema(self) -> dict:
        """Returns an OpenAPI-compatible JSON schema for the tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema(),
            },
        }

    def _get_parameters_schema(self) -> dict:
        """Returns the JSON schema for the tool's parameters."""
        raise NotImplementedError(
            "_get_parameters_schema must be implemented by Tool subclasses. This method should "
            "return an OpenAPI-compatible JSON schema dict describing the tool's input parameters. "
            "The schema should include parameter names, types, descriptions, and any validation rules."
        )


class FunctionTool(Tool):
    """Tool implementation that wraps a function"""

    func: Callable
    name: str
    description: str
    _input_schema: Type[BaseModel]

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # First process all the validation and setup
        processed_name = name or func.__name__

        # Validate function has type annotations
        if not all(get_type_hints(func).values()):
            raise ValueError(
                f"Tool '{processed_name}' must have complete type annotations for all parameters "
                "and return value."
            )

        # Parse the docstring and get description
        docstring = inspect.getdoc(func)
        if not docstring:
            raise ValueError(
                f"Tool '{processed_name}' must have a docstring with Google-style formatting."
            )

        doc_info = parse_docstring(docstring)
        processed_description = description or doc_info.description

        # Ensure we have parameter documentation
        if not doc_info.params:
            raise ValueError(
                f"Tool '{processed_name}' must have documented parameters in Google-style format. "
                "Example:\n    Args:\n        param_name: description"
            )

        # Validate all parameters are documented
        sig_params = set(inspect.signature(func).parameters.keys())
        doc_params = set(doc_info.params.keys())
        if sig_params != doc_params:
            missing = sig_params - doc_params
            extra = doc_params - sig_params
            raise ValueError(
                f"Tool '{processed_name}' parameter documentation mismatch. "
                f"Missing docs for: {missing if missing else 'none'}. "
                f"Extra docs for: {extra if extra else 'none'}."
            )

        # Create the input schema
        processed_input_schema = self._create_schema_from_function(
            func, doc_info.params
        )

        # Now call parent class constructor with processed values
        super().__init__(
            func=func,
            name=processed_name,
            description=processed_description,
            _input_schema=processed_input_schema,
        )

    @staticmethod
    def _create_schema_from_function(
        func: Callable, param_descriptions: dict[str, str]
    ) -> Type[BaseModel]:
        """Creates a Pydantic model from function signature and parsed docstring"""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        fields = {}
        for name, param in sig.parameters.items():
            fields[name] = (
                type_hints.get(name, Any),
                Field(description=param_descriptions.get(name, f"Parameter: {name}")),
            )

        return create_model(f"{func.__name__.title()}Inputs", **fields)

    def __call__(self, **kwargs) -> Any:
        """Execute the tool's function with validated inputs"""
        validated_inputs = self._input_schema(**kwargs)
        return self.func(**validated_inputs.model_dump())

    def _get_parameters_schema(self) -> dict:
        """Returns the JSON schema for the tool's parameters."""
        return self._input_schema.model_json_schema()


@mlflow.trace(span_type="FUNCTION")
def execute_function(tool, args):
    result = tool(**args)
    return json.dumps(result)
