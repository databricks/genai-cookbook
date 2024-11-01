# Databricks notebook source
# MAGIC %md
# MAGIC # RAG only Agent using MLflow SDK
# MAGIC
# MAGIC In this notebook, we construct an Agent that always uses a Retriever tool.  The Agent is encapsulated in a MLflow PyFunc class called `RAGAgent()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents databricks-vectorsearch mlflow pydantic
# dbutils.library.restartPython()

# COMMAND ----------
import sys
# Add the parent directory to the path so we can import the `utils` modules
sys.path.append("../..")


import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import asdict, dataclass
import mlflow
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, ChatCompletionResponse, ChainCompletionChoice, Message
from mlflow.deployments import get_deploy_client

from utils.agents.vector_search import VectorSearchRetriever, VectorSearchRetrieverConfig
from utils.agents.config import load_first_yaml_file
from utils.agents.config import RAGConfig
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Agent

# COMMAND ----------

class RAGAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that only includes an LLM

    If no explicit configuration is provided, the Agent will attempt to load the configuration from the following locations:
    """
    def __init__(self):
        config_paths = [
            "../../configs/agent_model_config.yaml",
            "./configs/agent_model_config.yaml",
        ]
        rag_config_yml = load_first_yaml_file(config_paths)
        self.config = RAGConfig.parse_obj(yaml.safe_load(rag_config_yml))
        retriever_config = VectorSearchRetrieverConfig.parse_obj(self.config.vector_search_retriever_config)
        self.model_serving_client = get_deploy_client("databricks")

        # Load the retriever
        self.retriever = VectorSearchRetriever(
            config=retriever_config
        )

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
        system_prompt = self.config.llm_config.llm_system_prompt_template
        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": user_query},
            ],
        )

        model_response = response.choices[0]["message"]["content"]

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
        
        return model_response.choices[0]["message"]["content"]

    def chat_completion(self, messages: List[Dict[str, str]]):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # Trace the call to Model Serving
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
        return traced_create(endpoint=endpoint_name, inputs=inputs)

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
    agent = RAGAgent()
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
    agent.load_context(None)
    response = agent.predict(
        model_input=input_example
    )

