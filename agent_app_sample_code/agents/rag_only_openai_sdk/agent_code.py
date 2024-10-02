# Databricks notebook source
# MAGIC %md
# MAGIC # RAG only Agent
# MAGIC
# MAGIC In this notebook, we construct an Agent that always uses a Retriever tool.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents openai databricks-vectorsearch mlflow pydantic
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import asdict, dataclass
import mlflow
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, ChatCompletionResponse, ChainCompletionChoice, Message
# from openai import OpenAI
from mlflow.deployments import get_deploy_client

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retriever

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
        # print(self.config)
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
                    # print(score)
                    i = 0
                    for field in item[0:-1]:
                        # print(field + "--")
                        metadata[column_names[i]["name"]] = field
                        i = i + 1
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

class RAGAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that only includes an LLM
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

        # Load the retriever
        self.retriever = VectorSearchRetriever(
            self.config.get("retriever_config")
        )


    # def load_context(self, context):
    #     # OpenAI client used to query Databricks Chat Completion endpoint
    #     # self.model_serving_client = OpenAI(
    #     #     api_key=os.environ.get("DB_TOKEN"),
    #     #     base_url=str(os.environ.get("DB_WORKSPACE_URL")) + "/serving-endpoints",
    #     # )

        

    @mlflow.trace(name="chain", span_type="CHAIN")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = self.get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = self.extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = self.extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Retrieve docs
        # If there is chat history, re-write the user's query based on that history
        if len(self.chat_history) > 0:
            vs_query = self.query_rewrite(user_query, self.chat_history)
        else:
            vs_query = user_query

        context = self.retriever(vs_query)

        
        ##############################################################################
        # Generate Answer
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")
        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": user_query},
            ],
        )
        
        # TODO: make error handling more robust
        
        # model_response = response.choices[0].message.content #openai
        model_response = response.choices[0]["message"]["content"] #mlflow

        # 'content' is required to comply with the schema of the Agent, see https://docs.databricks.com/en/generative-ai/create-log-agent.html#input-schema-for-the-rag-agent
        return asdict(StringResponse(model_response))

    @mlflow.trace(span_type="PARSER")
    def query_rewrite(self, query, chat_history) -> str:
        ############
        # Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
        ############
        query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

        Chat history: {chat_history}

        Question: {question}"""

        chat_history_formatted = self.format_chat_history(chat_history)

        prompt = query_rewrite_template.format(question=query, chat_history=chat_history_formatted)

        model_response = self.chat_completion(messages=[{"role": "user", "content": prompt}])
        
        return model_response.choices[0]["message"]["content"] #mlflow
        #return model_response.choices[0].message.content # openai

    def chat_completion(self, messages: List[Dict[str, str]]):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # Trace the call to Model Serving - openai
        # traced_create = mlflow.trace(
        #     self.model_serving_client.chat.completions.create,
        #     name="chat_completions_api",
        #     span_type="CHAT_MODEL",
        # )

        # Trace the call to Model Serving - mlflow version 
        traced_create = mlflow.trace(
            self.model_serving_client.predict,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Call LLM 
        inputs = {
            "messages": messages,
            **llm_options
        }
        return traced_create(endpoint=endpoint_name, inputs=inputs) #mlflow
        #return traced_create(model=endpoint_name, messages=messages, **llm_options) #openai
    
    @mlflow.trace(span_type="PARSER")
    def get_messages_array(
        self, model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        if type(model_input) == ChatCompletionRequest:
            return model_input.messages
        elif type(model_input) == dict:
            return model_input.get("messages")
        elif type(model_input) == pd.DataFrame:
            return model_input.iloc[0].to_dict().get("messages")
        
    @mlflow.trace(span_type="PARSER")
    def extract_user_query_string(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> str:
        """
        Extracts user query string from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            User query string.
        """

        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        if isinstance(chat_messages_array[-1], dict):
            return chat_messages_array[-1]["content"]
        elif isinstance(chat_messages_array[-1], Message):
            return chat_messages_array[-1].content
        else:
            return chat_messages_array[-1]

    @mlflow.trace(span_type="PARSER")
    def extract_chat_history(
        self, chat_messages_array: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Extracts the chat history from the chat messages array.

        Args:
            chat_messages_array: Array of chat messages.

        Returns:
            The chat history.
        """
        # Convert DataFrame to dict
        if isinstance(chat_messages_array, pd.Series):
            chat_messages_array = chat_messages_array.tolist()

        # Dictionary, return as is
        if isinstance(chat_messages_array[0], dict):
            return chat_messages_array[:-1]  # return all messages except the last one
        # MLflow Message, convert to Dictionary
        elif isinstance(chat_messages_array[0], Message):
            new_array = []
            for message in chat_messages_array[:-1]:
                new_array.append(asdict(message))
            return new_array
        else:
            raise ValueError(
                "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
            )

    @mlflow.trace(span_type="PARSER")
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Formats the chat history into a string.

        Args:
            chat_history: List of chat messages.

        Returns:
            Formatted chat history string.
        """
        if not chat_history:
            return ""

        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append(f"User: {message['content']}")

            # this logic ignores assistant messages that are just about tool calling and have no user facing content
            elif message["role"] == "assistant" and message.get("content"):
                formatted_history.append(f"Assistant: {message['content']}")

        return "\n".join(formatted_history)

set_model(RAGAgent())

# COMMAND ----------

# Set to False for logging, True for when iterating on code in this notebook 
debug = False

# To run this code, you will need to first run 02_agent to dump the configuration to a YAML file this notebook can load.
if debug:
    chain = RAGAgent()
    input_example = {
        "messages": [
            {
                "role": "user",
                "content": "what is rag?",
            },
            {
                "role": "assistant",
                "content": "its raggy",
            },
            {
                "role": "user",
                "content": "so how do i use it?",
            },
        ]
    }
    chain.load_context(None)
    response = chain.predict(
        model_input=input_example
    )

