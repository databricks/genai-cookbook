from cookbook.config.tools.base import Tool


from mlflow.models.resources import DatabricksResource
from pydantic import BaseModel, Field, create_model
from unitycatalog.ai.core.utils.docstring_utils import parse_docstring


import inspect
from typing import Any, Callable, List, Type, get_type_hints


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

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        return []
