# Databricks notebook source
# MAGIC %md
# MAGIC # Function Calling Agent w/ Retriever 
# MAGIC
# MAGIC In this notebook, we construct the Agent with a Retriever tool.  This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents databricks-vectorsearch mlflow pydantic
# dbutils.library.restartPython()

# COMMAND ----------

import json
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from mlflow.deployments import get_deploy_client
import os
from utils.agents.agent_utils import get_messages_array, extract_user_query_string, extract_chat_history

# COMMAND ----------

# MAGIC %md ##### Retriever tool

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]
    type: str


class VectorSearchRetriever:
    """
    Class using Databricks Vector Search to retrieve relevant documents.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_search_client = VectorSearchClient(disable_notice=True)
        self.vector_search_index = self.vector_search_client.get_index(
            index_name=self.config.get("vector_search_index")
        )

        vector_search_schema = self.config.get("vector_search_schema")
        mlflow.models.set_retriever_schema(
            primary_key=vector_search_schema.get("primary_key"),
            text_column=vector_search_schema.get("chunk_text"),
            doc_uri=vector_search_schema.get("document_uri"),
        )

    @mlflow.trace(span_type="TOOL", name="VectorSearchRetriever")
    def __call__(self, query: str) -> str:
        results = self.similarity_search(query)

        context = ""
        for result in results:
            formatted_chunk = self.config.get("chunk_template").format(
                chunk_text=result.get("page_content"),
                metadata=json.dumps(result.get("metadata")),
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

        traced_search = mlflow.trace(
            self.vector_search_index.similarity_search,
            name="vector_search.similarity_search",
        )

        vector_search_schema = self.config.get("vector_search_schema")
        additional_metadata_columns = (
            vector_search_schema.get("additional_metadata_columns") or []
        )

        columns = [
            vector_search_schema.get("primary_key"),
            vector_search_schema.get("chunk_text"),
            vector_search_schema.get("document_uri"),
        ] + additional_metadata_columns

        # de-duplicate
        columns = list(set(columns))

        if filters is None:
            results = traced_search(
                query_text=query,
                columns=columns,
                **self.config.get("vector_search_parameters"),
            )
        else:
            results = traced_search(
                query_text=query,
                filters=filters,
                columns=columns,
                **self.config.get("vector_search_parameters"),
            )

        # We turn the config into a dict and pass it here
        vector_search_threshold = self.config.get("vector_search_threshold")
        documents = self.convert_vector_search_to_documents(
            results, vector_search_threshold
        )

        return [asdict(doc) for doc in documents]

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
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]
                    del metadata[
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]

                    doc = Document(
                        page_content=page_content, metadata=metadata, type="Document"
                    )
                    docs.append(doc)

        return docs

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Agent

# COMMAND ----------

class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools
    """
    def __init__(self, agent_config: dict = None):
        self.__agent_config = agent_config
        try:
          self.config = mlflow.models.ModelConfig(development_config=self.__agent_config)
        except Exception as e:
            self.config = None
        if self.config is None:
            try:
                self.config = mlflow.models.ModelConfig(development_config="../../configs/agent_model_config.yaml")
            except Exception as e:
                self.config = None
        if self.config is None:
            self.config = mlflow.models.ModelConfig(development_config="./configs/agent_model_config.yaml")
            
        self.model_serving_client = get_deploy_client("databricks")

        # Initialize the tools
        self.tool_functions = {}
        self.tool_json_schemas =[]
        for tool in self.config.get("llm_config").get("tools"):
            # 1 Instantiate the tool's class w/ by passing the tool's config to it
            # 2 Store the instantiated tool to use later
            self.tool_functions[tool.get("tool_name")] = globals()[tool.get("tool_class_name")](config=tool)
            self.tool_json_schemas.append(tool.get("tool_input_json_schema"))

        self.chat_history = []

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Call LLM

        # messages to send the model
        # For models with shorter context length, you will need to trim this to ensure it fits within the model's context length
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": "user", "content": user_query}]
        )

        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self.recursively_call_and_run_tools(messages=messages)

        # If your front end keeps of converastion history and automatically appends the bot's response to the messages history, remove this line.
        messages_log_with_tool_calls.append(model_response.choices[0]["message"])

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        
        return {
            "content": model_response.choices[0]["message"]["content"], #mlflow client
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.

            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        for _ in range(max_iter):
            response = self.chat_completion(messages=messages)
            assistant_message = response.choices[0]["message"]
            tool_calls = assistant_message.get('tool_calls')
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:
                function = tool_call['function']
                args = json.loads(function['arguments'])
                result = self.execute_function(function['name'], args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": result,
                }
                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.copy()
            del assistant_message_dict["content"]
            messages = (
                messages
                + [
                    assistant_message_dict,
                ]
                + tool_messages
            )
        raise "ERROR: max iter reached"

    @mlflow.trace(span_type="FUNCTION")
    def execute_function(self, function_name, args):
        the_function = self.tool_functions[function_name]
        result = the_function(**args)
        return result

    def chat_completion(self, messages: List[Dict[str, str]]):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # Trace the call to Model Serving - mlflow version
        traced_create = mlflow.trace(
            self.model_serving_client.predict,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Get all tools
        tools = self.tool_json_schemas

        inputs = {
            "messages": messages,
            "tools": tools,
            **llm_options,
        }

        # Use the traced_create to make the prediction
        return traced_create(
            endpoint=endpoint_name,
            inputs=inputs,
        )


set_model(FunctionCallingAgent())
