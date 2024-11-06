# Databricks notebook source
# MAGIC %md
# MAGIC # RAG only Agent using LangChain SDK
# MAGIC
# MAGIC In this notebook, we construct an Agent that always uses a Retriever tool.  The Agent is a LangChain LCEL object called `agent`.

# COMMAND ----------

# # # If running this notebook by itself, uncomment these.
# %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 pydantic
# dbutils.library.restartPython()

# COMMAND ----------
import sys

# Add the parent directory to the path so we can import the `utils` modules
sys.path.append("../..")

from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.messages import HumanMessage, AIMessage
from utils.agents.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)
from utils.agents.rag_only_agent import RAGConfig
from utils.agents.yaml_loader import load_first_yaml_file
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC #### Agent

# COMMAND ----------

## Enable MLflow Tracing
mlflow.langchain.autolog()

# Load the chain's configuration
# This logic allows the code to run from this notebook OR the 02_agent notebook.
config_paths = [
    "../../configs/agent_model_config.yaml",
    "./configs/agent_model_config.yaml",
]
rag_config_yml = load_first_yaml_file(config_paths)
rag_config = RAGConfig.parse_obj(yaml.safe_load(rag_config_yml))

retriever_config = rag_config.vector_search_retriever_config
llm_config = rag_config.llm_config

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(index_name=retriever_config.vector_search_index)
vector_search_schema = retriever_config.vector_search_schema

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=vector_search_schema.chunk_text,
    columns=[
        vector_search_schema.primary_key,
        vector_search_schema.chunk_text,
        vector_search_schema.document_uri,
    ],
).as_retriever(search_kwargs=retriever_config.vector_search_parameters)

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key=vector_search_schema.primary_key,
    text_column=vector_search_schema.chunk_text,
    doc_uri=vector_search_schema.document_uri,  # Review App uses `doc_uri` to display chunks from the same document in a single view
)


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = retriever_config.chunk_template
    chunk_contents = [
        # Change the params here if you adjust the `chunk_template`
        chunk_template.format(
            chunk_text=d.page_content,
            metadata=d.metadata,
        )
        for d in docs
    ]
    return "".join(chunk_contents)


############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            llm_config.llm_system_prompt_template,
        ),
        # If there is history, provide it.
        # Note: This chain does not compress the history, so very long converastions can overflow the context window.
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        # User's most current question
        ("user", "{question}"),
    ]
)


# Format the converastion history to fit into the prompt template above.
def format_chat_history_for_prompt(chat_messages_array):
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(
                    HumanMessage(content=chat_message["content"])
                )
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(
                    AIMessage(content=chat_message["content"])
                )
    return formatted_chat_history


############
# Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
############
query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

query_rewrite_prompt = PromptTemplate(
    template=query_rewrite_template,
    input_variables=["chat_history", "question"],
)


############
# FM for generation
############
model = ChatDatabricks(
    endpoint=llm_config.llm_endpoint_name,
    extra_params=llm_config.llm_parameters,
)

############
# RAG Agent
############
agent = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages")
        | RunnableLambda(format_chat_history_for_prompt),
    }
    | RunnablePassthrough()
    | {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["chat_history"]) > 0,
                query_rewrite_prompt | model | StrOutputParser(),
            ),
            itemgetter("question"),
        )
        | vector_search_as_retriever
        | RunnableLambda(format_context),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

mlflow.models.set_model(model=agent)

# COMMAND ----------

# Set to False for logging, True for when iterating on code in this notebook
debug = False

# To run this code, you will need to first run 02_agent to dump the configuration to a YAML file this notebook can load.
if debug:
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
    response = agent.invoke(input_example)
