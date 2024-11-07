import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass, field
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from databricks.sdk import WorkspaceClient
import os
from cookbook.agents.utils.chat import (
    remove_message_keys_with_null_values,
)
from cookbook.agents.utils.load_config import load_config
from cookbook.config.agents.multi_agent_supervisor import (
    MultiAgentSupervisorConfig,
    WORKER_PROMPT_TEMPLATE,
    ROUTING_FUNCTION_NAME,
    CONVERSATION_HISTORY_THINKING_PARAM,
    WORKER_CAPABILITIES_THINKING_PARAM,
    NEXT_WORKER_OR_FINISH_PARAM,
    FINISH_ROUTE_NAME,
    SUPERVISOR_ROUTE_NAME,
)
from cookbook.agents.utils.chat import get_messages_array
from cookbook.agents.utils.playground_parser import (
    convert_messages_to_playground_tool_display_strings,
)
import importlib
import logging

# logging.basicConfig(level=logging.INFO)

from mlflow.entities import Trace
import mlflow.deployments

from enum import Enum


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


MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "multi_agent_supervisor_config.yaml"


class ConversationStatus(Enum):
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class SupervisorState:
    """Tracks essential conversation state"""

    chat_history: List[Dict[str, str]] = field(default_factory=list)
    current_route: str = SUPERVISOR_ROUTE_NAME
    number_of_workers_called: int = 0
    num_messages_at_start: int = 0
    status: ConversationStatus = ConversationStatus.ACTIVE
    error: Optional[str] = None

    @mlflow.trace()
    def append_new_message_to_history(self, message: Dict[str, str]) -> None:
        span = mlflow.get_current_active_span()
        span.set_inputs({"message": message})
        message_with_no_null_values_for_keys = remove_message_keys_with_null_values(
            message
        )
        self.chat_history.append(message_with_no_null_values_for_keys)
        span.set_outputs(self.chat_history)

    @mlflow.trace()
    def replace_chat_history(self, new_chat_history: List[Dict[str, str]]) -> None:
        span = mlflow.get_current_active_span()
        span.set_inputs(
            {
                "new_chat_history": new_chat_history,
                "current_chat_history": self.chat_history,
            }
        )
        messages_with_no_null_values_for_keys = []
        for message in new_chat_history:
            messages_with_no_null_values_for_keys.append(
                remove_message_keys_with_null_values(message)
            )
        self.chat_history = messages_with_no_null_values_for_keys.copy()
        span.set_outputs(self.chat_history)

    def finish(self, error: Optional[str] = None) -> None:
        self.status = ConversationStatus.ERROR if error else ConversationStatus.FINISHED
        self.error = error


class MultiAgentSupervisor(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def __init__(
        self, agent_config: Optional[Union[MultiAgentSupervisorConfig, str]] = None
    ):
        logging.info("Initializing MultiAgentSupervisor")
        # Initialize core configuration
        self._initialize_config(agent_config)

        # Initialize clients
        self._initialize_model_serving_clients()

        # Set up agents and routing
        self._initialize_supervised_agents()

        # Set up prompts and tools
        self._initialize_supervisor_prompts_and_tools()

        # Initialize state
        self.state = None  # Will be initialized per conversation
        logging.info("Initialized MultiAgentSupervisor")

    def _initialize_config(
        self, agent_config: Optional[Union[MultiAgentSupervisorConfig, str]]
    ):
        """Initialize and validate the agent configuration"""
        self.agent_config = load_config(
            agent_config=agent_config,
            default_config_file_name=MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            raise ValueError("No agent config found")

        # Set core configuration values
        self.MAX_LOOPS = self.agent_config.max_workers_called
        self.debug = self.agent_config.playground_debug_mode
        logging.info("Initialized agent config")

    def _initialize_model_serving_clients(self):
        """Initialize API clients for model serving"""
        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # used for calling the child agent's deployments
        self.mlflow_serving_client = mlflow.deployments.get_deploy_client("databricks")
        logging.info("Initialized model serving clients")

    def _initialize_supervised_agents(self):
        """Initialize the agent registry and capabilities"""
        # Initialize base agents (finish and supervisor)
        self.agents = {
            FINISH_ROUTE_NAME: {
                "agent_fn": self.finish_agent,
                "agent_description": "End the conversation, returning the last message to the user.",
            },
            SUPERVISOR_ROUTE_NAME: {
                "agent_fn": self.supervisor_agent,
                "agent_description": "Controls the conversation, deciding which Agent to use next. It only makes decisions about which agent to call, and does not respond to the user.",
            },
        }

        # Add configured worker agents
        if self.agent_config.agent_loading_mode == "model_serving":
            # using the model serving endpoints of the agents
            for agent in self.agent_config.agents:
                self.agents[agent.name] = {
                    "agent_description": agent.description,
                    "endpoint_name": agent.endpoint_name,
                }
        elif self.agent_config.agent_loading_mode == "local":
            # using the local agent classes
            for agent in self.agent_config.agents:
                # load the agent class
                module_name, class_name = agent.agent_class_path.rsplit(".", 1)

                module = importlib.import_module(module_name)
                # Load the Agent class, which will be a PyFunc
                agent_class_obj = getattr(module, class_name)
                self.agents[agent.name] = {
                    "agent_description": agent.description,
                    "agent_pyfunc_instance": agent_class_obj(
                        agent_config=agent.agent_config
                    ),  # instantiate the PyFunc
                }
                logging.info(f"Loaded agent: {agent.name}")
        else:
            raise ValueError(
                f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
            )

    def _initialize_supervisor_prompts_and_tools(self):
        """Initialize prompts and function calling tools"""
        # Create agents string for system prompt
        agents_info = [
            WORKER_PROMPT_TEMPLATE.format(
                worker_name=key, worker_description=value["agent_description"]
            )
            for key, value in self.agents.items()
        ]
        workers_names_and_descriptions = "".join(agents_info)

        # Update system prompt with template variables
        self.system_prompt = self.agent_config.supervisor_system_prompt.format(
            ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
            CONVERSATION_HISTORY_THINKING_PARAM=CONVERSATION_HISTORY_THINKING_PARAM,
            WORKER_CAPABILITIES_THINKING_PARAM=WORKER_CAPABILITIES_THINKING_PARAM,
            NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
            FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
            workers_names_and_descriptions=workers_names_and_descriptions,
        )

        # Initialize routing function schema
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

    @mlflow.trace()
    def route_response(self, next_route):
        # print("selected next: " + next_route)
        self.state.current_route = next_route
        return self.call_next_route()

    @mlflow.trace()
    def call_next_route(self):
        span = mlflow.get_current_active_span()
        span.set_attributes(asdict(self.state))
        logging.info(f"Calling next route: {self.state.current_route}")
        if self.state.current_route == FINISH_ROUTE_NAME:
            return self.finish_agent()
        elif self.state.current_route == SUPERVISOR_ROUTE_NAME:
            return self.supervisor_agent()
        else:
            agent_output = self.call_agent(self.state.current_route)
            # TODO: the agents by default return the enitre message history in their response, but they should provide a way to return only the new messages
            self.state.replace_chat_history(agent_output["messages"])
            self.state.current_route = SUPERVISOR_ROUTE_NAME
            return self.call_next_route()

    @mlflow.trace()
    def append_to_chat_history(self, message: Dict[str, Any]):
        self.state.append_new_message_to_history(message)
        mlflow.get_current_active_span().set_outputs(self.state.chat_history)
        # return self.state.messages  # hack to show in mlflow trace

    @mlflow.trace()
    def end_early(self):
        return self.route_response(FINISH_ROUTE_NAME)

    @mlflow.trace(span_type="AGENT")
    def supervisor_agent(self):

        if self.state.number_of_workers_called >= self.agent_config.max_workers_called:
            # finish early
            logging.warning("Finishing early due to exceeding max iterations.")
            return self.end_early()

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.state.chat_history
            + [
                {
                    "role": "user",
                    "content": self.agent_config.supervisor_user_prompt.format(
                        worker_names_with_finish=list(self.agents.keys())
                        + [FINISH_ROUTE_NAME],
                        NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
                        ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
                        FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
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
                    self.state.number_of_workers_called += 1
                    return self.route_response(
                        next_route=args[NEXT_WORKER_OR_FINISH_PARAM]
                    )
                else:
                    logging.error(
                        f"Supervisor LLM failed to call the {ROUTING_FUNCTION_NAME}(...) function to determine the next step, so we will default to finishing.  It tried to call `{function.name}` with args `{function.arguments}`."
                    )
                    self.state.current_route = FINISH_ROUTE_NAME
                    return self.call_next_route()
        else:
            logging.error(
                f"Supervisor LLM failed to choose a tool at all, so we will default to finishing.  It said `{assistant_message}`."
            )
            self.state.current_route = FINISH_ROUTE_NAME
            return self.call_next_route()

    @mlflow.trace()
    def call_agent(self, agent_name):
        # print(agent_name)
        if self.agent_config.agent_loading_mode == "model_serving":
            endpoint_name = self.agents.get(agent_name).get("endpoint_name")
            if endpoint_name:
                # this request will grab the mlflow trace from the endpoint
                request = {
                    "databricks_options": {"return_trace": True},
                    "messages": self.state.chat_history,
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
                raise ValueError(f"Invalid agent selected: {self.state.current_route}")
        elif self.agent_config.agent_loading_mode == "local":
            agent_pyfunc_instance = self.agents.get(agent_name).get(
                "agent_pyfunc_instance"
            )
            if agent_pyfunc_instance:
                request = {
                    # "databricks_options": {"return_trace": True},
                    "messages": self.state.chat_history,
                }
                return agent_pyfunc_instance.predict(model_input=request)
            else:
                raise ValueError(f"Invalid agent selected: {self.state.current_route}")
        else:
            raise ValueError(
                f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
            )

    @mlflow.trace(span_type="PARSER")
    def finish_agent(self):
        # Update to use config debug mode
        if self.agent_config.playground_debug_mode is True:
            return {
                "response": (
                    self.state.chat_history[-1]["content"]
                    if self.state.chat_history
                    else ""
                ),
                "messages": self.state.chat_history,
                # only parse the new messages we added into playground format
                "content": convert_messages_to_playground_tool_display_strings(
                    self.state.chat_history[self.state.num_messages_at_start :]
                ),
            }
        else:
            return {
                "content": (
                    self.state.chat_history[-1]["content"]
                    if self.state.chat_history
                    else ""
                ),
                "messages": self.state.chat_history,
            }

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        try:
            messages = get_messages_array(model_input)
            self.state = SupervisorState()
            self.state.replace_chat_history(messages)
            self.state.num_messages_at_start = len(messages)
            result = self.call_next_route()
            self.state.finish()
            return result

        except Exception as e:
            self.state.finish(error=str(e))
            raise

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


# tell MLflow logging where to find the agent's code
set_model(MultiAgentSupervisor)


# IMPORTANT: set this to False before logging the model to MLflow
debug = (
    __name__ == "__main__"
)  ## run in debug mode if being called by > python function_calling_agent.py


if debug:

    agent = MultiAgentSupervisor(agent_config=MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME)

    vibe_check_query = {
        "messages": [
            # {"role": "user", "content": f"how does the CoolTech Elite 5500 work?"},
            # {"role": "user", "content": f"calculate the value of 2+2?"},
            {
                "role": "user",
                "content": f"How does account age affect the likelihood of churn?",
            },
        ]
    }

    output = agent.predict(model_input=vibe_check_query)
    print(output["content"])

    input_2 = output["messages"].copy()
    input_2.append(
        {
            "role": "user",
            "content": f"who is the user most likely to do this?",
        },
    )

    output_2 = agent.predict(model_input={"messages": input_2})
    print(output_2["content"])

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
