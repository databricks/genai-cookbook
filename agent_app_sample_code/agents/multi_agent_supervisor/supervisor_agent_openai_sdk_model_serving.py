# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Agent Orchestrator
# MAGIC
# MAGIC In this notebook, we construct a multi-agent orchestrator using the supervisor pattern with the OpenAI SDK and Python code. This Agent is encapsulated in a MLflow PyFunc class called `MultiAgentSupervisor()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install -U -r requirements.txt
# dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from databricks.sdk import WorkspaceClient
import os
from utils.agents.utils import load_config
from utils.agents.multi_agent import (
    MultiAgentSupervisorConfig,
    WORKER_PROMPT_TEMPLATE,
    ROUTING_FUNCTION_NAME,
    CONVERSATION_HISTORY_THINKING_PARAM,
    WORKER_CAPABILITIES_THINKING_PARAM,
    NEXT_WORKER_OR_FINISH_PARAM,
    FINISH_ROUTE_NAME,
    SUPERVISOR_ROUTE_NAME,
)

import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)

from mlflow.entities import Trace
import mlflow.deployments


# COMMAND ----------

# MAGIC %md
# MAGIC # MultiAgentSupervisor Flow Documentation
# MAGIC
# MAGIC The MultiAgentSupervisor implements a supervisor pattern to orchestrate multiple specialized agents.
# MAGIC Here's how the control flow works:
# MAGIC
# MAGIC ## Entry Point
# MAGIC The flow begins when `predict()` is called with a user message:
# MAGIC 1. Initializes chat history with the user's message
# MAGIC 2. Sets `next_route` to "SUPERVISOR"
# MAGIC 3. Enters the main orchestration loop via `call_next_route()`
# MAGIC
# MAGIC ## Main Orchestration Loop
# MAGIC The loop consists of these key functions working together:
# MAGIC
# MAGIC ### `call_next_route()`
# MAGIC Central routing function that:
# MAGIC - Checks `next_route` and directs to one of:
# MAGIC   - `finish_agent()` if route is "FINISH"
# MAGIC   - `supervisor_agent()` if route is "SUPERVISOR"
# MAGIC   - `call_agent(agent_name)` for any other agent
# MAGIC - After a worker agent completes, automatically sets `next_route` back to "SUPERVISOR"
# MAGIC - Recursively calls itself to continue the loop
# MAGIC
# MAGIC ### `supervisor_agent()`
# MAGIC The orchestrator that:
# MAGIC 1. Assembles the supervisor LLM prompt:
# MAGIC    - Supervisor system prompt
# MAGIC    - Chat history (from the users' conversation)
# MAGIC    - Supervisor routing prompt
# MAGIC 2. Calls the LLM with function calling enabled
# MAGIC 3. Processes the LLM's routing decision by parsing the function call to populate `next_route`
# MAGIC 4. Calls `route_response()` with the next agent to invoke
# MAGIC
# MAGIC ### `route_response()`
# MAGIC Simple connector that:
# MAGIC 1. Inspects `next_route` to determine the next step take
# MAGIC 2. Triggers `call_next_route()` to execute that route
# MAGIC
# MAGIC ### `finish_agent()`
# MAGIC Exit point that:
# MAGIC 1. Returns the final response and full conversation history
# MAGIC 2. Optionally formats the conversation in playground style if debug mode is enabled
# MAGIC
# MAGIC ## Loop Termination
# MAGIC The loop continues until either:
# MAGIC 1. The supervisor selects "FINISH" as the next route
# MAGIC 2. The maximum number of iterations (`max_workers_called`) is reached
# MAGIC 3. An error occurs
# MAGIC
# MAGIC Each agent's response is added to the chat history, allowing the supervisor to maintain context and make informed routing decisions throughout the conversation.


# COMMAND ----------

CONFIG_FILE_NAME = "multi_agent_supervisor_config.yaml"


class MultiAgentSupervisor(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def __init__(
        self, agent_config: Optional[Union[MultiAgentSupervisorConfig, str]] = None
    ):
        self.agent_config: MultiAgentSupervisorConfig = load_config(
            agent_config=agent_config, default_config_file_name=CONFIG_FILE_NAME
        )
        if not self.agent_config:
            raise ValueError("No agent config found")

        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # used for calling the child agent's deployments
        self.mlflow_serving_client = mlflow.deployments.get_deploy_client("databricks")

        # internal agents.  finish agent is just a Callable with logic to return the last message to the user / format messages in playground style.
        self.agents = {
            FINISH_ROUTE_NAME: {
                "agent_fn": self.finish_agent,
                "agent_description": "End the converastion, returning the last message to the user.",
            },
            SUPERVISOR_ROUTE_NAME: {
                "agent_fn": self.supervisor_agent,
                "agent_description": "Controls the conversation, deciding which Agent to use next.  It only makes decisions about which agent to call, and does not respond to the user.",
            },
        }

        # initialize each child agent & where to find it
        for agent in self.agent_config.agents:
            self.agents[agent.name] = {
                "agent_description": agent.description,
                "endpoint_name": agent.endpoint_name,
            }

        # Create agents string for system prompt's `agent_config.supervisor_system_prompt`
        agents_info = [
            WORKER_PROMPT_TEMPLATE.format(
                worker_name=key, worker_description=value["agent_description"]
            )
            for key, value in self.agents.items()
        ]
        workers_names_and_descriptions = "".join(agents_info)

        # Update to use config values instead of hardcoded constants
        self.MAX_LOOPS = self.agent_config.max_workers_called
        self.debug = self.agent_config.playground_debug_mode

        # Update system prompt with template variables
        self.system_prompt = self.agent_config.supervisor_system_prompt.format(
            ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
            CONVERSATION_HISTORY_THINKING_PARAM=CONVERSATION_HISTORY_THINKING_PARAM,
            WORKER_CAPABILITIES_THINKING_PARAM=WORKER_CAPABILITIES_THINKING_PARAM,
            NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
            FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
            workers_names_and_descriptions=workers_names_and_descriptions,
        )

        # Create the supervisor routing function
        self.route_function = {
            "type": "function",
            "function": {
                "name": ROUTING_FUNCTION_NAME,
                "description": "Route the conversation by providing your thinking and next worker selection.",
                "parameters": {
                    "properties": {
                        CONVERSATION_HISTORY_THINKING_PARAM: {"type": "string"},
                        WORKER_CAPABILITIES_THINKING_PARAM: {"type": "string"},
                        NEXT_WORKER_OR_FINISH_PARAM: {
                            "enum": list(self.agents.keys()),
                            "type": "string",
                        },
                    },
                    "required": [
                        CONVERSATION_HISTORY_THINKING_PARAM,
                        WORKER_CAPABILITIES_THINKING_PARAM,
                        NEXT_WORKER_OR_FINISH_PARAM,
                    ],
                    "type": "object",
                },
            },
        }
        self.tool_json_schemas = [self.route_function]

        # empty state
        self.chat_history = []
        self.next_route = SUPERVISOR_ROUTE_NAME

        # track how many supervsior <> agent loops so we can cap it
        self.num_loops = 0

        # track how many messages came in the history, so when we dump playground formatted messages out, we don't re-dump the history
        self.num_in_history = None

    @mlflow.trace()
    def route_response(self, next_route):
        # print("selected next: " + next_route)
        self.next_route = next_route
        return self.call_next_route()

    @mlflow.trace()
    def call_next_route(self):
        logging.info(f"Calling next route: {self.next_route}")
        if self.next_route == FINISH_ROUTE_NAME:
            return self.finish_agent()
        elif self.next_route == SUPERVISOR_ROUTE_NAME:
            return self.supervisor_agent()
        else:
            agent_output = self.call_agent(self.next_route)
            self.chat_history = agent_output["messages"]
            # print(agent_output["content"])
            agent_func = self.agents.get(self.next_route)
            self.next_route = SUPERVISOR_ROUTE_NAME
            return self.call_next_route()

    @mlflow.trace()
    def append_to_chat_history(self, message: Dict[str, Any]):
        self.chat_history.append(message)
        return self.chat_history  # hack to show in mlflow trace

    @mlflow.trace()
    def end_early(self):
        return self.route_response("FINISH")

    @mlflow.trace(span_type="AGENT")
    def supervisor_agent(self):

        if self.num_loops >= self.agent_config.max_workers_called:
            # finish early
            logging.warning("Finishing early due to exceeding max iterations.")
            return self.end_early()

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.chat_history
            + [
                {
                    "role": "user",
                    "content": self.agent_config.supervisor_user_prompt.format(
                        worker_names_with_finish=list(self.agents.keys())
                        + [FINISH_ROUTE_NAME],
                        NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
                        ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
                    ),
                }
            ]
        )

        response = self.chat_completion(messages=messages, tools=True)
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call.function
                args = json.loads(function.arguments)
                if function.name == ROUTING_FUNCTION_NAME:
                    self.num_loops += 1
                    return self.route_response(
                        next_route=args[NEXT_WORKER_OR_FINISH_PARAM]
                    )
                else:
                    logging.error(
                        f"Supervisor LLM failed to call the {ROUTING_FUNCTION_NAME}(...) function to determine the next step, so we will default to finishing.  It tried to call `{function.name}` with args `{function.arguments}`."
                    )
                    self.next_route = FINISH_ROUTE_NAME
                    return self.call_next_route()
        else:
            logging.error(
                f"Supervisor LLM failed to choose a tool at all, so we will default to finishing.  It said `{assistant_message}`."
            )
            self.next_route = FINISH_ROUTE_NAME
            return self.call_next_route()

    @mlflow.trace()
    def call_agent(self, agent_name):
        # print(agent_name)
        endpoint_name = self.agents.get(agent_name).get("endpoint_name")
        if endpoint_name:
            # this request will grab the mlflow trace from the endpoint
            request = {
                "databricks_options": {"return_trace": True},
                "messages": self.chat_history,
            }
            completion = self.mlflow_serving_client.predict(
                endpoint=endpoint_name, inputs=request
            )

            logging.info(f"Called agent: {agent_name}")
            logging.info(f"Got response agent: {completion}")

            # Add the trace from model serving API call to the active trace
            if trace := completion.pop("databricks_output", {}).get("trace"):
                trace = Trace.from_dict(trace)
                mlflow.add_trace(trace)

            return completion
        else:
            error_message = {
                "role": "assistant",
                "content": f"ERROR: This agent does not exist.  Please select one of: {list(self.agents.keys())}",
                "name": SUPERVISOR_ROUTE_NAME,
            }
            raise ValueError(f"Invalid agent selected: {self.next_route}")

    @mlflow.trace(span_type="PARSER")
    def finish_agent(self):
        # Update to use config debug mode
        if self.agent_config.playground_debug_mode is True:
            return {
                "response": (
                    self.chat_history[-1]["content"] if self.chat_history else ""
                ),
                "messages": self.chat_history,
                # only parse the new messages we added into playground format
                "content": self.stringify_messages(
                    self.chat_history[self.num_in_history :]
                ),
            }
        else:
            return {
                "content": (
                    self.chat_history[-1]["content"] if self.chat_history else ""
                ),
                "messages": self.chat_history,
                # "": self.stringify_messages(self.chat_history),
            }

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        # Extract `messages` key from the `model_input`
        messages = self.get_messages_array(model_input)

        # Set initial chat_history
        self.chat_history = self.convert_messages_to_open_ai_format(messages)

        self.num_in_history = len(self.chat_history)

        # reset state
        self.next_route = SUPERVISOR_ROUTE_NAME
        self.num_loops = 0

        # enter the supervisor loop
        return self.call_next_route()

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.agent_config.llm_endpoint_name
        llm_options = self.agent_config.llm_parameters.model_dump()

        # # Trace the call to Model Serving - openai versio
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Openai - start
        if tools:
            return traced_create(
                model=endpoint_name,
                messages=messages,
                tools=self.tool_json_schemas,
                parallel_tool_calls=False,
                **llm_options,
            )
        else:
            return traced_create(model=endpoint_name, messages=messages, **llm_options)
        # Openai - end

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
    def convert_messages_to_open_ai_format(
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
            return chat_messages_array  # return all messages except the last one
        # MLflow Message, convert to Dictionary
        elif isinstance(chat_messages_array[0], Message):
            new_array = []
            for message in chat_messages_array:
                new_array.append(asdict(message))
            return new_array
        else:
            raise ValueError(
                "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
            )

    # helpers for playground & review app formatting of tool calls
    @mlflow.trace(span_type="PARSER")
    def stringify_messages(self, messages: List[Dict[str, str]]) -> str:
        output = ""
        for msg in messages:  # ignore first user input
            if msg["role"] == "assistant" and msg.get("tool_calls"):  # tool call
                for tool_call in msg["tool_calls"]:
                    output += self.stringify_tool_call(tool_call)
                # output += f"<uc_function_call>{json.dumps(msg, indent=2)}</uc_function_call>"
            elif msg["role"] == "tool":  # tool response
                output += self.stringify_tool_result(msg)
                # output += f"<uc_function_result>{json.dumps(msg, indent=2)}</uc_function_result>"
            else:
                output += msg["content"] if msg["content"] != None else ""
        return output

    @mlflow.trace(span_type="PARSER")
    def stringify_tool_call(self, tool_call) -> str:
        try:
            function = tool_call["function"]
            args_dict = json.loads(function["arguments"])
            request = {
                "id": tool_call["id"],
                "name": function["name"],
                "arguments": json.dumps(args_dict),
            }

            return f"<uc_function_call>{json.dumps(request)}</uc_function_call>"

        except Exception as e:
            print("Failed to stringify tool call: ", e)
            return str(tool_call)

    @mlflow.trace(span_type="PARSER")
    def stringify_tool_result(self, tool_msg) -> str:
        try:

            result = json.dumps(
                {"id": tool_msg["tool_call_id"], "content": tool_msg["content"]}
            )
            return f"<uc_function_result>{result}</uc_function_result>"
        except Exception as e:
            print("Failed to stringify tool result:", e)
            return str(tool_msg)


# tell MLflow logging where to find the agent's code
set_model(MultiAgentSupervisor())

# COMMAND ----------

debug = True

# COMMAND ----------

# DBTITLE 1,debugging code


if debug:

    agent = MultiAgentSupervisor()

    vibe_check_query = {
        "messages": [
            {"role": "user", "content": f"what issues do we have with returned items?"},
            # {"role": "user", "content": f"calculate the value of 2+2?"},
            # {
            #     "role": "user",
            #     "content": f"How does account age affect the likelihood of churn?",
            # },
        ]
    }

    output = agent.predict(model_input=vibe_check_query)
    print(output)

#     input_2 = output["messages"].copy()
#     input_2.append(
#         {
#             "role": "user",
#             "content": f"did user 8e753fa6-2464-4354-887c-a25ace971a7e experience these?",
#         },
#     )

#     output_2 = agent.predict(model_input={"messages": input_2})

# # COMMAND ----------

# if debug:
#     agent = MultiAgentSupervisor(agent_config="supervisor_config.yml")
#     vibe_check_query = {
#         "messages": [
#             # {"role": "user", "content": f"What is agent evaluation?"},
#             # {"role": "user", "content": f"What users have churned?"},
#             {
#                 "role": "user",
#                 "content": f"What is the capacity of the BrewMaster Elite 3000 coffee maker?",
#             },
#             # {"role": "user", "content": f"calculate the value of 2+2?"},
#             # {
#             #     "role": "user",
#             #     "content": f"did user 8e753fa6-2464-4354-887c-a25ace971a7e experience any issues?",
#             # },
#         ]
#     }

#     output = agent.predict(model_input=vibe_check_query)
#     # print(output)
