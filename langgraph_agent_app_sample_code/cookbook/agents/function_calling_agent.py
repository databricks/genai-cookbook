# In this file, we construct a function-calling Agent with a Retriever tool using MLflow + langgraph.

import logging
import json
from dataclasses import asdict
from functools import reduce
from typing import Iterator, Dict, List, Optional, Union, Any

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
from mlflow.models.rag_signatures import (
    ChatCompletionResponse,
    ChainCompletionChoice,
    Message,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
)
from mlflow.models.resources import (
    DatabricksResource,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksFunction,
)
from pydantic import BaseModel
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit

from cookbook.agents.utils.load_config import load_config
from cookbook.config.agents.function_calling_agent import FunctionCallingAgentConfig
from cookbook.config.shared.llm import LLMConfig
from cookbook.config.shared.tool import (
    ToolConfig,
    VectorSearchToolConfig,
    UCFunctionToolConfig,
)

logging.basicConfig(level=logging.INFO)


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
                    chunk_text=doc.page_content,
                )
                for doc in relevant_docs
            ]
            return "".join(chunk_contents)

        return search_product_docs
    elif type(tool_config) == UCFunctionToolConfig:
        client = DatabricksFunctionClient()
        toolkit = UCFunctionToolkit(
            client=client, function_names=[tool_config.function_name]
        )
        return toolkit.tools[-1]
    else:
        raise ValueError(f"Unknown tool type: {tool_config.type}")

def create_chat_completion_response(message: Message) -> Dict:
    return asdict(ChatCompletionResponse(
        choices=[ChainCompletionChoice(message=message)],
    ))

def stringify_tool_call(tool_call: Dict[str, Any]) -> str:
    """Convert a raw tool call into a formatted string that the playground UI expects"""
    try:
        request = json.dumps(
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "arguments": str(tool_call.get("args", {})),
            },
            indent=2,
        )
        return f"<uc_function_call>{request}</uc_function_call>"
    except:
        # for non UC functions, return the string representation of tool calls
        # you can modify this to return a different format
        return str(tool_call)


def stringify_tool_result(tool_msg: ToolMessage) -> str:
    """Convert a ToolMessage into a formatted string that the playground UI expects"""
    try:
        result = json.dumps(
            {"id": tool_msg.tool_call_id, "content": tool_msg.content}, indent=2
        )
        return f"<uc_function_result>{result}</uc_function_result>"
    except:
        # for non UC functions, return the string representation of tool message
        # you can modify this to return a different format
        return str(tool_msg)


def parse_message(msg) -> Message:
    """Parse different message types into their string representations"""
    # tool call result
    if isinstance(msg, ToolMessage):
        return Message(role="tool", content=stringify_tool_result(msg))
    # tool call
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_call_results = [stringify_tool_call(call) for call in msg.tool_calls]
        return Message(role="system", content="".join(tool_call_results))
    # normal HumanMessage or AIMessage (reasoning or final answer)
    elif isinstance(msg, AIMessage):
        return Message(role="system", content=msg.content)
    elif isinstance(msg, HumanMessage):
        return Message(role="user", content=msg.content)
    else:
        print(f"Unexpected message type: {type(msg)}")
        return Message(role="unknown", content=str(msg))


def wrap_output(stream: Iterator[MessageLikeRepresentation]) -> Iterator[Dict]:
    """
    Process and yield formatted outputs from the message stream.
    The invoke and stream langchain functions produce different output formats.
    This function handles both cases.
    """
    for event in stream:
        # the agent was called with invoke()
        if "messages" in event:
            messages = event["messages"]
            # output_content = ""
            # for msg in event["messages"]:
            #     output_content += parse_message(msg) + "\n\n"
            yield create_chat_completion_response(parse_message(messages[-1]))
        # the agent was called with stream()
        else:
            for node in event:
                for key, messages in event[node].items():
                    if isinstance(messages, list):
                        for msg in messages:
                            yield create_chat_completion_response(parse_message(msg))
                    else:
                        print(
                            "Unexpected value {messages} for key {key}. Expected a list of `MessageLikeRepresentation`'s"
                        )
                        yield create_chat_completion_response(Message(content=str(messages)))


def create_resource_dependency(config: BaseModel) -> List[DatabricksResource]:
    if isinstance(config, LLMConfig):
        return [DatabricksServingEndpoint(endpoint_name=config.llm_endpoint_name)]
    elif isinstance(config, VectorSearchToolConfig):
        return [
            DatabricksVectorSearchIndex(index_name=config.index_name),
            DatabricksServingEndpoint(config.embedding_endpoint_name),
        ]
    elif isinstance(config, UCFunctionToolConfig):
        return [DatabricksFunction(function_name=config.function_name)]
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def get_resource_dependencies(
    agent_config: FunctionCallingAgentConfig,
) -> List[DatabricksResource]:
    configs = [agent_config.llm_config] + agent_config.tool_configs
    dependencies = reduce(lambda x, y: x + y, map(create_resource_dependency, configs))
    return dependencies


def create_function_calling_agent(
    agent_config: Optional[Union[FunctionCallingAgentConfig, str]] = None
) -> RunnableSequence:
    if not agent_config:
        raise (
            f"No agent config found.  If you are in your local development environment, make sure you either [1] are calling init(agent_config=...) with either an instance of FunctionCallingAgentConfig or the full path to a YAML config file or [2] have a YAML config file saved at {{your_project_root_folder}}/configs/{FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME}."
        )

    tool_configs = agent_config.tool_configs
    tools = list(map(create_tool, tool_configs))

    chat_model = ChatDatabricks(
        endpoint=agent_config.llm_config.llm_endpoint_name,
        temerature=agent_config.llm_config.llm_parameters.temperature,
        max_tokens=agent_config.llm_config.llm_parameters.max_tokens,
    )

    react_agent = create_react_agent(
        chat_model,
        tools,
        messages_modifier=agent_config.llm_config.llm_system_prompt_template,
    ) | RunnableGenerator(wrap_output)

    logging.info("Successfully loaded agent config in __init__.")

    return react_agent


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