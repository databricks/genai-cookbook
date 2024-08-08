# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Dict, List, Optional

import mlflow
import yaml
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatMessage, ChatResponse, ChatChoice, TokenUsageStats
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# NOTE: this must be commented out when deploying the agent

# Get the API endpoint and token for the current notebook context
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Set these as environment variables
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# COMMAND ----------

# Load the chain config
model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

databricks_resources = model_config.get("databricks_resources")
retriever_config = model_config.get("retriever_config")
llm_config = model_config.get("llm_config")

# COMMAND ----------

class VectorSearchRetriever:
    def __init__(self, config):
        self.config = config
        self.vs_client = VectorSearchClient(disable_notice=True)
        self.vs_index = self.vs_client.get_index(
            endpoint_name=self.config["databricks_resources"]["vector_search_endpoint_name"],
            index_name=self.config["retriever_config"]["vector_search_index"],
        )

    @mlflow.trace(span_type="RETRIEVER")
    def vector_search(self, query: str) -> List[Dict[str, Any]]:
        results = self.vs_index.similarity_search(
            query_text=query,
            columns=[
                self.config["retriever_config"]["schema"]["primary_key"],
                self.config["retriever_config"]["schema"]["chunk_text"],
                self.config["retriever_config"]["schema"]["document_uri"],
            ],
            num_results=self.config["retriever_config"]["parameters"]["k"],
        )
        
        documents = [
            {
                "page_content": result[1],
                "metadata": {
                    self.config["retriever_config"]["schema"]["document_uri"]: result[2],
                    self.config["retriever_config"]["schema"]["primary_key"]: result[0],
                }
            }
            for result in results["result"]["data_array"]
        ]
        
        return documents

# COMMAND ----------

class RagChain(ChatModel):
    def __init__(self):
        self.config = None
        self.vector_search_retriever = None

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        if context is not None and context.artifacts is not None:
            config_path = mlflow.artifacts.download_artifacts(context.artifacts["config_path"])
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "databricks_resources": databricks_resources,
                "retriever_config": retriever_config,
                "llm_config": llm_config,
            }
        
        if not self.config:
            raise ValueError("No configuration provided")

        self.vector_search_retriever = VectorSearchRetriever(self.config)

        # Set retriever schema for MLflow
        mlflow.models.set_retriever_schema(
            primary_key=self.config["retriever_config"]["schema"]["primary_key"],
            text_column=self.config["retriever_config"]["schema"]["chunk_text"],
            doc_uri=self.config["retriever_config"]["schema"]["document_uri"],
        )

    @property
    def openai_client(self):
        # create the OpenAI client on-demand - necessary for pickling
        return OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN"),
            base_url=os.environ.get("DATABRICKS_HOST") + "/serving-endpoints",
        )

    def predict(self, context: mlflow.pyfunc.model.PythonModelContext, messages: List[Dict[str, str]]) -> ChatResponse:
        return self.run_chain(messages)

    @mlflow.trace(name="rag_chain", span_type="CHAIN")
    def run_chain(self, messages: List[Dict[str, str]]) -> ChatResponse:
        user_query = self.extract_user_query_string(messages)
        chat_history = self.extract_chat_history(messages)
        
        rewritten_query = self.rewrite_query(user_query, chat_history)
        retrieved_docs = self.vector_search_retriever.vector_search(rewritten_query)
        
        context = self.format_context(retrieved_docs)
        formatted_chat_history = self.format_chat_history_for_prompt(messages)
        
        response = self.generate_answer(user_query, context, formatted_chat_history)
        return self.convert_chat_response(response)

    @mlflow.trace(span_type="PARSER")
    def extract_user_query_string(self, chat_messages_array):
        return chat_messages_array[-1]["content"]

    @mlflow.trace(span_type="PARSER")
    def extract_chat_history(self, chat_messages_array):
        return chat_messages_array[:-1]

    @mlflow.trace(span_type="LLM")
    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        if not chat_history:
            return query

        query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

        Chat history: {chat_history}

        Question: {question}"""

        response = self.openai_client.chat.completions.create(
            model=self.config["databricks_resources"]["llm_endpoint_name"],
            messages=[
                {"role": "system", "content": query_rewrite_template},
                {"role": "user", "content": f"Chat history: {json.dumps(chat_history)}\n\nQuestion: {query}"},
            ],
            **self.config["llm_config"]["llm_parameters"],
        )
        return response.choices[0].message.content.strip()

    @mlflow.trace(span_type="PARSER")
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        chunk_template = self.config["retriever_config"]["chunk_template"]
        formatted_chunks = [
            chunk_template.format(
                chunk_text=doc["page_content"],
                document_uri=doc["metadata"][self.config["retriever_config"]["schema"]["document_uri"]],
            )
            for doc in retrieved_docs
        ]
        return "".join(formatted_chunks)

    @mlflow.trace(span_type="PARSER")
    def format_chat_history_for_prompt(self, chat_messages_array):
        history = self.extract_chat_history(chat_messages_array)
        formatted_chat_history = []
        if len(history) > 0:
            for chat_message in history:
                if chat_message["role"] == "user":
                    formatted_chat_history.append({"role": "user", "content": chat_message["content"]})
                elif chat_message["role"] == "assistant":
                    formatted_chat_history.append({"role": "assistant", "content": chat_message["content"]})
        return formatted_chat_history

    @mlflow.trace(span_type="LLM")
    def generate_answer(self, query: str, context: str, formatted_chat_history: List[Dict[str, str]]) -> Any:
        messages = [
            {"role": "system", 
             "content": self.config["llm_config"]["llm_system_prompt_template"].format(context=context)},
        ]
        messages.extend(formatted_chat_history)
        messages.append({"role": "user", "content": query})

        response = self.openai_client.chat.completions.create(
            model=self.config["databricks_resources"]["llm_endpoint_name"],
            messages=messages,
            **self.config["llm_config"]["llm_parameters"],
        )
        return response

    @mlflow.trace(span_type="PARSER")
    def convert_chat_response(self, response: Any) -> ChatResponse:
        return ChatResponse(
            id=response.id,
            model=response.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.choices[0].message.content),
                    finish_reason="stop",
                )
            ],
            usage=TokenUsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the chain to use for evaluation and deployment.  
# This is required to log this chain to MLflow with `mlflow.pyfunc.log_model(...)`.
mlflow.models.set_model(model=RagChain())        

# COMMAND ----------

# Initialize RagChain class
rag_chain = RagChain()
rag_chain.load_context(None) 

# Example input
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG stands for Retrieval-Augmented Generation...",
        },
        {
            "role": "user",
            "content": "How does it work?",
        },
    ]
}

# Invoke the chain
response = rag_chain.predict(None, input_example["messages"])
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
