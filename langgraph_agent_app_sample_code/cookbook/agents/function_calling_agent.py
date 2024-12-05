# In this file, we construct a function-calling Agent with a Retriever tool using MLflow + langgraph.

import logging
from dataclasses import asdict
from functools import reduce
from typing import Iterator, Dict, List

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch
from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
)
from langchain_core.runnables import RunnableGenerator
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tools import tool, Tool
from langgraph.prebuilt import create_react_agent
from mlflow.models import set_model
from mlflow.models.rag_signatures import ChatCompletionResponse, ChainCompletionChoice, \
    Message
from mlflow.models.resources import DatabricksResource, DatabricksServingEndpoint, \
    DatabricksVectorSearchIndex, DatabricksFunction
from pydantic import BaseModel
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit

from cookbook.agents.utils.load_config import load_config
from cookbook.config.agents.function_calling_agent import FunctionCallingAgentConfig
from cookbook.config.shared.llm import LLMConfig
from cookbook.config.shared.tool import ToolConfig, VectorSearchToolConfig, \
    UCFunctionToolConfig

FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "function_calling_agent_config.yaml"

def create_tool(tool_config: ToolConfig) -> Tool:
    if type(tool_config) == VectorSearchToolConfig:
        vector_search_as_retriever = DatabricksVectorSearch(
            endpoint=tool_config.endpoint,
            index_name=tool_config.index_name,
            columns=tool_config.columns,
        ).as_retriever(search_kwargs=tool_config.search_kwargs)

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

        return search_product_docs
    elif type(tool_config) == UCFunctionToolConfig:
        client = DatabricksFunctionClient()
        toolkit = UCFunctionToolkit(
            client=client,
            function_names=[tool_config.function_name]
        )
        return toolkit.tools[-1]
    else:
        raise ValueError(f"Unknown tool type: {tool_config.type}")


def create_chat_completion_response(content: str) -> Dict:
    return asdict(ChatCompletionResponse(
        choices=[ChainCompletionChoice(
            message=Message(role="assistant", content=content + "\n\n"))],
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
                            if isinstance(msg, AIMessage) and len(
                                    msg.tool_calls) == 0:  # final result
                                # Note: you can pass additional fields from your LangGraph nodes to the output here
                                yield create_chat_completion_response(
                                    content=msg.content)
                    else:
                        logging.warning(
                            "Unexpected value {messages} for key {key}. Expected a list of `MessageLikeRepresentation`'s")
                        yield create_chat_completion_response(content=str(messages))


def create_resource_dependency(config: BaseModel) -> List[DatabricksResource]:
    if type(config) == LLMConfig:
        return [DatabricksServingEndpoint(endpoint_name=config.llm_endpoint_name)]
    elif type(config) == VectorSearchToolConfig:
        return [DatabricksVectorSearchIndex(index_name=config.index_name), DatabricksServingEndpoint(config.embedding_endpoint)]
    elif type(config) == UCFunctionToolConfig:
        return [DatabricksFunction(function_name=config.function_name)]
    else:
        raise ValueError(f"Unknown config type: {config.type}")

def get_resource_dependencies(agent_config: FunctionCallingAgentConfig) -> List[DatabricksResource]:
    configs = [agent_config.llm_config] + agent_config.tool_configs
    dependencies = reduce(lambda x, y: x + y, map(create_resource_dependency, configs))
    return dependencies

def create_function_calling_agent(
        agent_conig: FunctionCallingAgentConfig) -> RunnableSequence:
    tool_configs = agent_conig.tool_configs
    tools = list(map(create_tool, tool_configs))

    chat_model = ChatDatabricks(endpoint=agent_conig.llm_config.llm_endpoint_name,
                                temerature=agent_conig.llm_config.llm_parameters.temperature,
                                max_tokens=agent_conig.llm_config.llm_parameters.max_tokens)

    react_agent = create_react_agent(
        chat_model,
        tools,
        messages_modifier=agent_conig.llm_config.llm_system_prompt_template
    ) | RunnableGenerator(wrap_output)

    return react_agent




logging.basicConfig(level=logging.INFO)

agent_conf = load_config(
            passed_agent_config=None,
            default_config_file_name=FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )

# tell MLflow logging where to find the agent's code
set_model(create_function_calling_agent(agent_conf))

# IMPORTANT: set this to False before logging the model to MLflow
debug = False

if debug:
    # logging.basicConfig(level=logging.INFO)
    # print(find_config_folder_location())
    # print(os.path.abspath(os.getcwd()))
    # mlflow.tracing.disable()
    agent = create_function_calling_agent()

    vibe_check_query = {
        "messages": [
            # {"role": "user", "content": f"what is agent evaluation?"},
            # {"role": "user", "content": f"How does the blender work?"},
            # {
            #     "role": "user",
            #     "content": f"find all docs from the section header 'Databricks documentation archive' or 'Work with files on Databricks'",
            # },
            {
                "role": "user",
                "content": "Translate the sku `OLD-abs-1234` to the new format",
            }
            # {
            #     "role": "user",
            #     "content": f"convert sku 'OLD-XXX-1234' to the new format",
            # },
            # {
            #     "role": "user",
            #     "content": f"what are recent customer issues?  what words appeared most frequently?",
            # },
        ]
    }

    output = agent.predict(model_input=vibe_check_query)
    print(output["content"])



