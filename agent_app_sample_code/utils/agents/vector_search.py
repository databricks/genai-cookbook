from dataclasses import dataclass, field, asdict

import mlflow
from mlflow.entities import Document
from mlflow.models.resources import (
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint,
)

import json

from typing import Literal, Optional, Any, Callable, Dict, List
from databricks.vector_search.client import VectorSearchClient

from pydantic import BaseModel
from utils.agents.tools import BaseTool

class RetrieverOutputSchema(BaseModel):
    """
    Configuration for the schema used in the retriever's response.

    Attributes:
        primary_key (str): The column name in the retriever's response referred to the unique key.
            If using Databricks vector search with delta sync, this should be the column of the delta table that acts as the primary key.
            Example: "chunk_id" - The column name in the retriever's response referred to the unique key.
        chunk_text (str): The column name in the retriever's response that contains the returned chunk.
            Example: "content_chunked" - The column name in the retriever's response that contains the returned chunk.
        document_uri (str): The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            Example: "doc_uri" - The URI of the chunk - displayed as the document ID in the Review App.
        additional_metadata_columns (List[str]): Additional metadata columns to present to the LLM.
            Example: [] - Additional columns to return from the vector database and present to the LLM.
    """

    # The column name in the retriever's response referred to the unique key
    # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
    primary_key: str
    # The column name in the retriever's response that contains the returned chunk.
    chunk_text: str
    # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
    document_uri: str
    # Additional metadata columns to present to the LLM.
    additional_metadata_columns: List[str]


class VectorSearchRetrieverInputSchema(BaseModel):
    """
    Configuration for the input schema (parameters) used in the retriever.

    Attributes:
        num_results (int): The number of chunks to return for each query.
            Example: 5 - Number of search results that the retriever returns.
        query_type (Literal['ann', 'hybrid']): The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid` (keyword + semantic similarity).
            Example: "ann" - Type of search: ann or hybrid.
    """

    # The number of chunks to return for each query.
    num_results: int
    # The type of search to use, either `ann` (semantic similarity with embeddings) or `hybrid`
    # (keyword + semantic similarity)
    query_type: Literal["ann", "hybrid"]

class VectorSearchRetrieverConfig(BaseModel):
    """
    Configuration for a Databricks Vector Search retriever, which can be used either deterministically in a
    fixed RAG chain or as a tool.

    Attributes:
        vector_search_index (str): Vector Search index that is created by the data pipeline.
            Example: datapipeline_output_config.vector_index - UC Vector Search index.
        vector_search_schema (RetrieverOutputSchema): Schema configuration for the retriever.
            Required by Agent Evaluation to:
            1. Enable the Review App to properly display retrieved chunks.
            2. Enable metrics / LLM judges to understand which fields to use to measure the retriever.
            Each is a column name within the `vector_search_index`.
        vector_search_threshold (float): Threshold for retrieved document similarity. Used to exclude results that are very dissimilar to the query.
            Example: 0.1 - 0 to 1, similarity threshold cut off for retrieved docs. Increase if the retriever is returning irrelevant content.
        chunk_template (str): Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.
            The f-string {chunk_text} and {metadata} can be used.
            Example: "Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n" - Prompt template used to format the retrieved information into {context} in `prompt_template`.
        prompt_template (str): Prompt template used to format all chunks for presentation to the LLM.
            The f-string {context} can be used.
        vector_search_parameters (VectorSearchRetrieverInputSchema): Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
            Parameters defined by Vector Search docs: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#query-a-vector-search-endpoint
    """

    # Vector Search index that is created by the data pipeline
    vector_search_index: str

    vector_search_schema: RetrieverOutputSchema

    # Threshold for retrieved document similarity.  Used to exclude results that are very dissimilar to the query.
    vector_search_threshold: float

    # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question.  The f-string {chunk_text} and {metadata} can be used.
    chunk_template: str

    # Prompt template used to format all chunks for presentation to the LLM.  The f-string {context} can be used.
    prompt_template: str

    # Extra parameters to pass to DatabricksVectorSearch.as_retriever(search_kwargs=parameters).
    vector_search_parameters: VectorSearchRetrieverInputSchema

    retriever_query_parameter_prompt: str = "The query to find documents for."

class VectorSearchRetriever(BaseModel):
    """
    Class using Databricks Vector Search to retrieve relevant documents.
    """
    config: VectorSearchRetrieverConfig

    def __init__(self, config: VectorSearchRetrieverConfig):
        super().__init__(config=config)
        vector_search_schema = self.config.vector_search_schema
        mlflow.models.set_retriever_schema(
            primary_key=vector_search_schema.primary_key,
            text_column=vector_search_schema.chunk_text,
            doc_uri=vector_search_schema.document_uri,
        )

    @mlflow.trace(span_type="TOOL", name="VectorSearchRetriever")
    def __call__(self, query: str) -> str:
        results = self.similarity_search(query)

        context = ""
        for result in results:
            formatted_chunk = self.config.chunk_template.format(
                chunk_text=result.page_content,
                metadata=json.dumps(result.metadata),
            )
            context += formatted_chunk

        return context.strip()

    @mlflow.trace(span_type="RETRIEVER")
    def similarity_search(
            self, query: str, filters: Dict[Any, Any] = None
    ) -> List[Document]:
        """
        Performs vector search to retrieve relevant chunks.

        Args:
            query: Search query.
            filters: Optional filters to apply to the search, must follow the Databricks Vector Search filter spec (https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#use-filters-on-queries)

        Returns:
            List of retrieved Documents.
        """

        vector_search_client = VectorSearchClient(disable_notice=True)
        vector_search_index = vector_search_client.get_index(
            index_name=self.config.vector_search_index
        )

        traced_search = mlflow.trace(
            vector_search_index.similarity_search,
            name="vector_search.similarity_search",
        )

        vector_search_schema = self.config.vector_search_schema
        additional_metadata_columns = (
                vector_search_schema.additional_metadata_columns or []
        )

        columns = [
                      vector_search_schema.primary_key,
                      vector_search_schema.chunk_text,
                      vector_search_schema.document_uri,
                  ] + additional_metadata_columns

        # de-duplicate
        columns = list(set(columns))

        if filters is None:
            results = traced_search(
                query_text=query,
                columns=columns,
                **self.config.vector_search_parameters.dict(),
            )
        else:
            results = traced_search(
                query_text=query,
                filters=filters,
                columns=columns,
                **self.config.vector_search_parameters.dict(),
            )

        # We turn the config into a dict and pass it here
        vector_search_threshold = self.config.vector_search_threshold
        return self.convert_vector_search_to_documents(
            results, vector_search_threshold
        )

    @mlflow.trace(span_type="PARSER")
    def convert_vector_search_to_documents(
            self, vs_results, vector_search_threshold
    ) -> List[Document]:
        column_names = []
        for column in vs_results["manifest"]["columns"]:
            column_names.append(column)

        docs = []
        if vs_results["result"]["row_count"] > 0:
            for item in vs_results["result"]["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata["similarity_score"] = score
                    for i, field in enumerate(item[0:-1]):
                        metadata[column_names[i]["name"]] = field
                    # put contents of the chunk into page_content
                    page_content = metadata[
                        self.config.vector_search_schema.chunk_text
                    ]
                    del metadata[
                        self.config.vector_search_schema.chunk_text
                    ]

                    doc = Document(
                        page_content=page_content, metadata=metadata
                    )
                    docs.append(doc)

        return docs

class VectorSearchRetrieverTool(BaseTool):

    vector_search_retriever: VectorSearchRetriever
    retriever_query_parameter_prompt: str

    def __init__(self, vector_search_retriever: VectorSearchRetriever, tool_description_prompt: str, tool_name: str, retriever_query_parameter_prompt: str):
        super().__init__(tool_description_prompt=tool_description_prompt, tool_name=tool_name, vector_search_retriever=vector_search_retriever, retriever_query_parameter_prompt=retriever_query_parameter_prompt)
        # self.vector_search_retriever = vector_search_retriever
        # self.retriever_query_parameter_prompt = retriever_query_parameter_prompt

    @property
    def tool_input_schema(self) -> dict:
        return {
            "properties": {
                "query": {
                    "default": None,
                    "description": self.retriever_query_parameter_prompt,
                    "type": "string",
                }
            },
            "type": "object",
        }

    def get_resource_dependencies(self):
        retriever_config = self.vector_search_retriever.config
        dependencies = [DatabricksVectorSearchIndex(index_name=retriever_config.vector_search_index)]

        index_embedding_model = (
            VectorSearchClient(disable_notice=True)
            .get_index(index_name=retriever_config.vector_search_index)
            .describe()
            .get("delta_sync_index_spec")
            .get("embedding_source_columns")[0]
            .get("embedding_model_endpoint_name")
        )
        if index_embedding_model is not None:
            dependencies.append(
                DatabricksServingEndpoint(endpoint_name=index_embedding_model),
            )
        else:
            raise Exception(f"Could not identify the embedding model endpoint resource for {retriever_config.vector_search_index}.  Please manually add the embedding model endpoint to `databricks_resources`.")
        return dependencies
