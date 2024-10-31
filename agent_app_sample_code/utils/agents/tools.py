import mlflow
from pydantic import computed_field, Field, BaseModel
from utils.agents.pydantic_utils import SerializableModel
@mlflow.trace(span_type="FUNCTION")
def execute_function(tool_functions, function_name, args):
    the_function = tool_functions[function_name]

    result = the_function(**args)
    return result


class BaseToolModel(SerializableModel):
    # A description of the documents in the index.  Used by the Agent to determine if this tool is relevant to the query.
    tool_description_prompt: str

    # The name of the tool.  Used by the Agent in conjunction with tool_description_prompt to determine if this tool is relevant to the query.
    tool_name: str

    @property
    def tool_input_schema(self) -> dict:
        raise Exception("tool_input_schema must be implemented by subclass")

    @computed_field
    def tool_input_json_schema(self) -> dict:
        tool_input_json_schema = self.tool_input_schema
        # del tool_input_json_schema["title"]
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description_prompt,
                "parameters": tool_input_json_schema,
            },
        }
