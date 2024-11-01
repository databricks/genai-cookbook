import mlflow
from typing import List
from pydantic import computed_field, Field, BaseModel
from utils.pydantic_utils import SerializableModel

@mlflow.trace(span_type="FUNCTION")
def execute_function(tool, args):
    result = tool(**args)
    return result


class BaseTool(SerializableModel):
    # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
    tool_description_prompt: str

    # The name of the tool.  Used by the Agent in conjunction with tool_description_prompt to determine if this tool is relevant to the query.
    tool_name: str

    @property
    def tool_input_schema(self) -> dict:
        raise Exception("tool_input_schema must be implemented by Tool subclasses, and must return an "
                        "OpenAPI-compatible dict of the tool's input parameters")

    def tool_input_json_schema(self) -> dict:
        tool_input_json_schema = self.tool_input_schema
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description_prompt,
                "parameters": tool_input_json_schema,
            },
        }

    def get_resource_dependencies(self) -> List[any]:
        return []
    
    def __call__(self, *args, **kwargs):
        raise Exception("Tool classes must define the implementation of the tool "
                        "(i.e. what happens when the tool is invoked by an LLM) in the "
                        "__call__ method")
