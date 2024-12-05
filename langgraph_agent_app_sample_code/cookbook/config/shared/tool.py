from pydantic import BaseModel
from enum import Enum


class ToolConfig(BaseModel):
    name: str
    description: str

class VectorSearchToolConfig(ToolConfig):
    endpoint: str
    index_name: str
    columns: str
    search_kwargs: dict
    embedding_endpoint_name: str

class UCFunctionToolConfig(ToolConfig):
    function_name: str