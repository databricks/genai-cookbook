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
    remove_tool_calls_from_messages,
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


MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "multi_agent_supervisor_config.yaml"

AGENT_RAW_OUTPUT_KEY = "raw_agent_output"
AGENT_NEW_MESSAGES_KEY = "new_messages"


@dataclass
class SupervisorState:
    """Tracks essential conversation state"""

    chat_history: List[Dict[str, str]] = field(default_factory=list)
    last_agent_called: str = ""
    number_of_supervisor_loops_completed: int = 0
    num_messages_at_start: int = 0
    # error: Optional[str] = None

    @mlflow.trace(span_type="FUNCTION", name="state.append_new_message_to_history")
    def append_new_message_to_history(self, message: Dict[str, str]) -> None:
        span = mlflow.get_current_active_span()
        span.set_inputs({"message": message})
        with mlflow.start_span(
            name="remove_message_keys_with_null_values"
        ) as span_inner:
            span_inner.set_inputs({"message": message})
            message_with_no_null_values_for_keys = remove_message_keys_with_null_values(
                message
            )
            span_inner.set_outputs(
                {
                    "message_with_no_null_values_for_keys": message_with_no_null_values_for_keys
                }
            )
        self.chat_history.append(message_with_no_null_values_for_keys)
        span.set_outputs(self.chat_history)

    @mlflow.trace(span_type="FUNCTION", name="state.overwrite_chat_history")
    def overwrite_chat_history(self, new_chat_history: List[Dict[str, str]]) -> None:
        span = mlflow.get_current_active_span()
        span.set_inputs(
            {
                "new_chat_history": new_chat_history,
                "current_chat_history": self.chat_history,
            }
        )
        messages_with_no_null_values_for_keys = []
        with mlflow.start_span(
            name="remove_message_keys_with_null_values"
        ) as span_inner:
            span_inner.set_inputs({"new_chat_history": new_chat_history})
            for message in new_chat_history:
                messages_with_no_null_values_for_keys.append(
                    remove_message_keys_with_null_values(message)
                )
            span_inner.set_outputs(
                {
                    "messages_with_no_null_values_for_keys": messages_with_no_null_values_for_keys
                }
            )
        self.chat_history = messages_with_no_null_values_for_keys.copy()
        span.set_outputs(self.chat_history)


class MultiAgentSupervisor(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def __init__(
        self, agent_config: Optional[Union[MultiAgentSupervisorConfig, str]] = None
    ):
        logging.info("Initializing MultiAgentSupervisor")

        # load the Agent's configuration. See load_config() for details.
        self.agent_config = load_config(
            passed_agent_config=agent_config,
            default_config_file_name=MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            raise ValueError(
                f"No agent config found.  If you are in your local development environment, make sure you either [1] are calling init(agent_config=...) with either an instance of MultiAgentSupervisorConfig or the full path to a YAML config file or [2] have a YAML config file saved at {{your_project_root_folder}}/configs/{MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME}."
            )
        else:
            logging.info("Successfully loaded agent config in __init__.")
            logging.info(f"Loaded config: {self.agent_config.model_dump()}")

        # Initialize clients
        self._initialize_model_serving_clients()

        # Set up agents and routing
        self._initialize_supervised_agents()

        # Set up prompts and tools
        self._initialize_supervisor_prompts_and_tools()

        # Initialize state
        self.state = None  # Will be initialized per conversation
        logging.info("Initialized MultiAgentSupervisor")

    def _initialize_model_serving_clients(self):
        """Initialize API clients for model serving"""
        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # used for calling the child agent's deployments
        self.mlflow_serving_client = mlflow.deployments.get_deploy_client("databricks")
        logging.info("Initialized model serving clients")

    def _initialize_supervised_agents(self):
        """Initialize the agent registry and capabilities"""
        self.agents = {}

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
        self.supervisor_system_prompt = (
            self.agent_config.supervisor_system_prompt.format(
                ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
                CONVERSATION_HISTORY_THINKING_PARAM=CONVERSATION_HISTORY_THINKING_PARAM,
                WORKER_CAPABILITIES_THINKING_PARAM=WORKER_CAPABILITIES_THINKING_PARAM,
                NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
                FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
                workers_names_and_descriptions=workers_names_and_descriptions,
            )
        )

        self.supervisor_user_prompt = self.agent_config.supervisor_user_prompt.format(
            worker_names_with_finish=list(self.agents.keys()) + [FINISH_ROUTE_NAME],
            NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
            ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
            FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
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

    @mlflow.trace(span_type="AGENT")
    def _get_supervisor_routing_decision(self, messages: List[Dict[str, str]]) -> str:

        supervisor_messages = (
            [{"role": "system", "content": self.supervisor_system_prompt}]
            + messages
            + [
                {
                    "role": "user",
                    "content": self.supervisor_user_prompt,
                }
            ]
        )

        response = self.chat_completion(messages=supervisor_messages, tools=True)
        supervisor_llm_response = response.choices[0].message
        supervisor_tool_calls = supervisor_llm_response.tool_calls

        if supervisor_tool_calls:
            for tool_call in supervisor_tool_calls:
                function = tool_call.function
                args = json.loads(function.arguments)
                if function.name == ROUTING_FUNCTION_NAME:
                    return args  # includes all keys from the function call
                else:
                    logging.error(
                        f"Supervisor LLM failed to call the {ROUTING_FUNCTION_NAME}(...) function to determine the next step, so we will default to finishing.  It tried to call `{function.name}` with args `{function.arguments}`."
                    )
                    return FINISH_ROUTE_NAME
        else:
            logging.error(
                f"Supervisor LLM failed to choose a tool at all, so we will default to finishing.  It said `{supervisor_llm_response}`."
            )
            return FINISH_ROUTE_NAME

    @mlflow.trace()
    def _call_supervised_agent(
        self, agent_name: str, input_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Calls a supervised agent and returns ONLY the new [messages] produced by that agent.
        """
        raw_agent_output = {}
        if self.agent_config.agent_loading_mode == "model_serving":
            endpoint_name = self.agents.get(agent_name).get("endpoint_name")
            if endpoint_name:
                # this request will grab the mlflow trace from the endpoint
                request = {
                    "databricks_options": {"return_trace": True},
                    "messages": input_messages.copy(),
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

                raw_agent_output = completion
            else:
                raise ValueError(f"Invalid agent selected: {agent_name}")
        elif self.agent_config.agent_loading_mode == "local":
            agent_pyfunc_instance = self.agents.get(agent_name).get(
                "agent_pyfunc_instance"
            )
            if agent_pyfunc_instance:
                request = {
                    # "databricks_options": {"return_trace": True},
                    "messages": input_messages.copy(),
                }
                raw_agent_output = agent_pyfunc_instance.predict(model_input=request)
            else:
                raise ValueError(f"Invalid agent selected: {agent_name}")
        else:
            raise ValueError(
                f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
            )

        # return only the net new messages produced by the agent
        agent_output_messages = raw_agent_output.get("messages", [])
        num_messages_previously = len(input_messages)
        num_messages_after_agent = len(agent_output_messages)
        if (
            num_messages_after_agent == 0
            or num_messages_after_agent == num_messages_previously
        ):
            raise Exception(
                f"Agent {agent_name} either returned no messages at all or returned the same number of messages it received, indicating it did not produce any new messages."
            )

        else:
            # Add the Agent's name to its messages
            new_messages = agent_output_messages[num_messages_previously:].copy()
            for new_message in new_messages:
                new_message["name"] = agent_name
            return {
                # agent's raw output
                AGENT_RAW_OUTPUT_KEY: raw_agent_output,
                # new messages produced by the agent
                AGENT_NEW_MESSAGES_KEY: new_messages,
            }

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        # try:
        # Initialize conversation state
        messages = get_messages_array(model_input)
        self.state = SupervisorState()
        self.state.overwrite_chat_history(messages)
        self.state.num_messages_at_start = len(messages)

        # Run the supervisor loop up to self.agent_config.max_workers_called times
        while (
            self.state.number_of_supervisor_loops_completed
            < self.agent_config.max_supervisor_loops
        ):
            with mlflow.start_span(name="supervisor_loop_iteration") as span:
                self.state.number_of_supervisor_loops_completed += 1

                chat_history_without_tool_calls = remove_tool_calls_from_messages(
                    self.state.chat_history
                )
                routing_function_output = self._get_supervisor_routing_decision(
                    chat_history_without_tool_calls
                )

                next_agent = routing_function_output.get(NEXT_WORKER_OR_FINISH_PARAM)
                span.set_inputs(
                    {
                        f"supervisor.{NEXT_WORKER_OR_FINISH_PARAM}": next_agent,
                        f"supervisor.{CONVERSATION_HISTORY_THINKING_PARAM}": routing_function_output.get(
                            CONVERSATION_HISTORY_THINKING_PARAM
                        ),
                        f"supervisor.{WORKER_CAPABILITIES_THINKING_PARAM}": routing_function_output.get(
                            WORKER_CAPABILITIES_THINKING_PARAM
                        ),
                        "state.number_of_workers_called": self.state.number_of_supervisor_loops_completed,
                        "state.chat_history": self.state.chat_history,
                        "chat_history_without_tool_calls": chat_history_without_tool_calls,
                    }
                )

                if next_agent is None:
                    logging.error(
                        f"Supervisor returned no next agent, so we will default to finishing."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": "Supervisor returned no next agent, so we will default to finishing.",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )
                    break
                if next_agent == FINISH_ROUTE_NAME:
                    logging.info(
                        f"Supervisor called {FINISH_ROUTE_NAME} after {self.state.number_of_supervisor_loops_completed} workers being called."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": "Supervisor selected it.",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )
                    break  # finish by exiting the while loop
                # prevent the supervisor from calling an agent multiple times in a row
                elif next_agent != self.state.last_agent_called:
                    # Call worker agent and update history
                    try:
                        agent_output = self._call_supervised_agent(
                            next_agent, chat_history_without_tool_calls
                        )
                        agent_new_messages = agent_output[AGENT_NEW_MESSAGES_KEY]
                        agent_raw_output = agent_output[AGENT_RAW_OUTPUT_KEY]

                        self.state.overwrite_chat_history(
                            self.state.chat_history + agent_new_messages
                        )
                        self.state.last_agent_called = next_agent
                        span.set_outputs(
                            {
                                "post_processed_decision": next_agent,
                                "post_processing_reason": "Supervisor selected it.",
                                "updated_chat_history": self.state.chat_history,
                                f"called_agent.{AGENT_NEW_MESSAGES_KEY}": agent_new_messages,
                                f"called_agent.{AGENT_RAW_OUTPUT_KEY}": agent_raw_output,
                            }
                        )

                    except ValueError as e:
                        logging.error(
                            f"Error calling agent {next_agent}: {e}.  We will default to finishing."
                        )
                        span.set_outputs(
                            {
                                "post_processed_decision": FINISH_ROUTE_NAME,
                                "post_processing_reason": "Supervisor selected an invalid agent, so defaulting to finishing.",
                                "updated_chat_history": self.state.chat_history,
                            }
                        )
                        break  # finish by exiting the while loop
                else:
                    logging.warning(
                        f"Supervisor called the same agent {next_agent} twice in a row.  We will default to finishing."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": f"Supervisor selected {next_agent} twice in a row, so business logic decided to finish instead.",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )
                    break  # finish by exiting the while loop

        # if the last message is not from the assistant, we need to add a fake assistant message
        # TODO: add the name of the supervisor agent here
        if self.state.chat_history[-1]["role"] != "assistant":
            logging.warning(
                "No assistant ended up replying, so we'll add an error response"
            )
            with mlflow.start_span(name="add_error_response_to_history") as span:
                span.set_inputs(
                    {
                        "state.chat_history": self.state.chat_history,
                    }
                )
                self.state.append_new_message_to_history(
                    {
                        "role": "assistant",
                        "content": self.agent_config.supervisor_error_response,
                        # "name": "supervisor",
                    }
                )
                span.set_outputs(
                    {
                        "updated_chat_history": self.state.chat_history,
                    }
                )

        # Return the resulting conversation back to the user
        with mlflow.start_span(name="return_conversation_to_user") as span:
            span.set_inputs(
                {
                    "state.chat_history": self.state.chat_history,
                    "agent_config.playground_debug_mode": self.agent_config.playground_debug_mode,
                }
            )
            if self.agent_config.playground_debug_mode is True:
                return_value = {
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
                span.set_outputs(return_value)
                return return_value
            else:
                return_value = {
                    "content": (
                        self.state.chat_history[-1]["content"]
                        if self.state.chat_history
                        else ""
                    ),
                    "messages": self.state.chat_history,
                }
                span.set_outputs(return_value)
                return return_value

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
set_model(MultiAgentSupervisor())


# IMPORTANT: set this to False before logging the model to MLflow
debug = False

if debug:

    # agent = MultiAgentSupervisor(agent_config=MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME)
    agent = MultiAgentSupervisor()

    vibe_check_query = {
        "messages": [
            {"role": "user", "content": f"how does the CoolTech Elite 5500 work?"},
            # {"role": "user", "content": f"calculate the value of 2+2?"},
            # {
            #     "role": "user",
            #     "content": f"How does account age affect the likelihood of churn?",
            # },
        ]
    }

    output = agent.predict(model_input=vibe_check_query)
    print(output["content"])
    # print(output)

    # input_2 = output["messages"].copy()
    # input_2.append(
    #     {
    #         "role": "user",
    #         "content": f"who is the user most likely to do this?",
    #         # "content": f"how do i turn it on?",
    #     },
    # )

    # output_2 = agent.predict(model_input={"messages": input_2})
    # print(output_2["content"])

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
