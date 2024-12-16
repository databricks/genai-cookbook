from pydantic import BaseModel
from cookbook.config import SerializableConfig
from typing import List

class ToolConfig(SerializableConfig):
    name: str
    description: str

class VectorSearchToolConfig(ToolConfig):
    endpoint: str
    index_name: str
    columns: List[str]
    search_kwargs: dict
    embedding_endpoint_name: str

class UCFunctionToolConfig(ToolConfig):
    function_name: str