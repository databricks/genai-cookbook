from mlflow.models import set_model
from unitycatalog.ai.core.client import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableGenerator
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
)
from langchain_core.tools import tool, Tool

from mlflow.langchain.output_parsers import ChatCompletionsOutputParser
from typing import Iterator, Dict, Any
from mlflow.models.rag_signatures import ChatCompletionResponse, ChainCompletionChoice, Message
from random import randint
from dataclasses import asdict
import logging
from databricks_langchain import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
import json
from databricks_langchain import ChatDatabricks
from langgraph.prebuilt import ToolNode

client = DatabricksFunctionClient()


CATALOG = "shared"  # Change me!
SCHEMA = "cookbook_local_test_udhay"  # Change me if you want

python_execution_function_name = f"{CATALOG}.{SCHEMA}.python_exec"


# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    endpoint="one-env-shared-endpoint-9",
    index_name=f"{CATALOG}.{SCHEMA}.test_product_docs_docs_chunked_index__v2",
    columns=["chunk_id", "content_chunked", "doc_uri"],
).as_retriever(search_kwargs={"k": 3})


@tool
def search_product_docs(question: str):
    """Use this tool to search for databricks product documentation."""
    relevant_docs = vector_search_as_retriever.get_relevant_documents(question)
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in relevant_docs
    ]
    return "".join(chunk_contents)


toolkit = UCFunctionToolkit(
    client=client,
    function_names=[python_execution_function_name]
)
python_exec_tool = toolkit.tools[-1]

tools = [search_product_docs, python_exec_tool]

tool_node = ToolNode(tools)


def create_chat_completion_response(content: str) -> Dict:
    return asdict(ChatCompletionResponse(
        choices=[ChainCompletionChoice(message=Message(role="assistant", content=content+"\n\n"))],
    ))

def wrap_output(stream: Iterator[MessageLikeRepresentation]) -> Iterator[Dict]:
    """
    Process and yield formatted outputs from the message stream.
    The invoke and stream langchain functions produce different output formats.
    This function handles both cases.
    """
    for event in stream:
        # the agent was called with invoke()
        if "messages" in event:
            output_content = ""
            for msg in event["messages"]:
                output_content += msg.content
            # Note: you can pass additional fields from your LangGraph nodes to the output here
            yield create_chat_completion_response(content=output_content)
        # the agent was called with stream()
        else:
            for node in event:
                for key, messages in event[node].items():
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, AIMessage) and len(msg.tool_calls) == 0: # final result
                            # Note: you can pass additional fields from your LangGraph nodes to the output here
                                yield create_chat_completion_response(content=msg.content)
                    else:
                        logging.warning("Unexpected value {messages} for key {key}. Expected a list of `MessageLikeRepresentation`'s")
                        yield create_chat_completion_response(content=str(messages))


chat_model = ChatDatabricks(endpoint="agents-demo-gpt4o-mini")

system_prompt = """
## Role
You are a helpful assistant that answers questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.

## Objective
Your goal is to provide accurate, relevant, and helpful response based solely on the outputs from these tools. You are concise and direct in your responses.

## Instructions
1. **Understand the Query**: Think step by step to analyze the user's question and determine the core need or problem. 

2. **Assess available tools**: Think step by step to consider each available tool and understand their capabilities in the context of the user's query.

3. **Select the appropriate tool(s) OR ask follow up questions**: Based on your understanding of the query and the tool descriptions, decide which tool(s) should be used to generate a response. If you do not have enough information to use the available tools to answer the question, ask the user follow up questions to refine their request.  If you do not have a relevant tool for a question or the outputs of the tools are not helpful, respond with: "I'm sorry, I can't help you with that."
""".strip()

react_agent = create_react_agent(
    chat_model,
    tools,
    messages_modifier=system_prompt
) | RunnableGenerator(wrap_output)
set_model(react_agent)
