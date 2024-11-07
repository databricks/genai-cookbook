from cookbook.tools import Tool


from mlflow.models.resources import DatabricksResource
from pydantic import BaseModel, Field, create_model
from unitycatalog.ai.core.utils.docstring_utils import parse_docstring
from typing import Optional

import inspect
from typing import Any, Callable, List, Type, get_type_hints
import importlib


class LocalFunctionTool(Tool):
    """Tool implementation that wraps a function"""

    # func: Callable
    func_path: str
    name: str
    description: str
    _input_schema: Type[BaseModel]

    def _process_function(
        self, func: Callable, name: Optional[str], description: Optional[str]
    ) -> tuple[str, str, Type[BaseModel]]:
        """Process a function to extract name, description and input schema.

        Args:
            func: The function to process
            name: Optional override for the function name
            description: Optional override for the function description

        Returns:
            Tuple of (processed_name, processed_description, processed_input_schema)
        """
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

        return processed_name, processed_description, processed_input_schema

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        *,
        func: Optional[Callable] = None,
        func_path: Optional[str] = None,
    ):
        if func is not None and func_path is not None:
            raise ValueError("Only one of func or func_path can be provided")

        if func is not None:
            # Process the function to get name, description and input schema
            processed_name, processed_description, processed_input_schema = (
                self._process_function(func, name, description)
            )

            # Serialize the function's location
            func_path = f"{func.__module__}.{func.__name__}"

            # Now call parent class constructor with processed values
            super().__init__(
                func_path=func_path,
                name=processed_name,
                description=processed_description,
            )

            self._input_schema = processed_input_schema

            self._loaded_callable = None
            self.load_func()
        elif func_path is not None:

            super().__init__(
                func_path=func_path,
                name=name,
                description=description,
                # _input_schema=None,
            )

            self._loaded_callable = None
            self.load_func()

            _, _, processed_input_schema = self._process_function(
                self._loaded_callable, name, description
            )

            self._input_schema = processed_input_schema

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

    def load_func(self):
        if self._loaded_callable is None:
            module_name, func_name = self.func_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self._loaded_callable = getattr(module, func_name)

    def __call__(self, **kwargs) -> Any:
        """Execute the tool's function with validated inputs"""
        self.load_func()
        validated_inputs = self._input_schema(**kwargs)
        return self._loaded_callable(**validated_inputs.model_dump())

    def _get_parameters_schema(self) -> dict:
        """Returns the JSON schema for the tool's parameters."""
        return self._input_schema.model_json_schema()

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        return []
