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


MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "multi_agent_supervisor_config.yaml"


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
        message_with_no_null_values_for_keys = remove_message_keys_with_null_values(
            message
        )
        self.chat_history.append(message_with_no_null_values_for_keys)
        span.set_outputs(self.chat_history)

    @mlflow.trace(span_type="FUNCTION", name="state.replace_chat_history")
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
                "agent_fn": None,
                "agent_description": "End the conversation, returning the last message to the user.",
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
    def _call_agent(
        self, agent_name: str, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        if self.agent_config.agent_loading_mode == "model_serving":
            endpoint_name = self.agents.get(agent_name).get("endpoint_name")
            if endpoint_name:
                # this request will grab the mlflow trace from the endpoint
                request = {
                    "databricks_options": {"return_trace": True},
                    "messages": messages,
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
                raise ValueError(f"Invalid agent selected: {agent_name}")
        elif self.agent_config.agent_loading_mode == "local":
            agent_pyfunc_instance = self.agents.get(agent_name).get(
                "agent_pyfunc_instance"
            )
            if agent_pyfunc_instance:
                request = {
                    # "databricks_options": {"return_trace": True},
                    "messages": messages,
                }
                return agent_pyfunc_instance.predict(model_input=request)
            else:
                raise ValueError(f"Invalid agent selected: {agent_name}")
        else:
            raise ValueError(
                f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
            )

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
        self.state.replace_chat_history(messages)
        self.state.num_messages_at_start = len(messages)

        # Run the supervisor loop up to self.agent_config.max_workers_called times
        while (
            self.state.number_of_supervisor_loops_completed
            < self.agent_config.max_supervisor_loops
        ):
            with mlflow.start_span(name="supervisor_loop_iteration") as span:
                self.state.number_of_supervisor_loops_completed += 1
                routing_function_output = self._get_supervisor_routing_decision(
                    self.state.chat_history
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
                    }
                )

                if next_agent is None:
                    logging.error(
                        f"Supervisor returned no next agent, so we will default to finishing."
                    )
                    next_agent = FINISH_ROUTE_NAME
                    break
                if next_agent == FINISH_ROUTE_NAME:
                    logging.info(
                        f"Supervisor called {FINISH_ROUTE_NAME} after {self.state.number_of_supervisor_loops_completed} workers being called."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": "Supervisor selected it & business logic agrees.",
                        }
                    )
                    break  # finish by exiting the while loop
                # prevent the supervisor from calling an agent multiple times in a row
                elif next_agent != self.state.last_agent_called:
                    # Call worker agent and update history
                    try:
                        agent_output = self._call_agent(
                            next_agent, self.state.chat_history
                        )
                        agent_produced_message_history = agent_output[
                            "messages"
                        ]  # includes the previous history too
                        num_messages_previously = len(self.state.chat_history)
                        num_messages_after_agent = len(agent_produced_message_history)
                        num_new_messages = (
                            num_messages_after_agent - num_messages_previously
                        )
                        message_history_updated_with_agent_name = (
                            agent_produced_message_history.copy()
                        )
                        # for each new message (the agent won't / shouldn't modify the previous history)
                        for new_message in message_history_updated_with_agent_name[
                            num_messages_previously:
                        ]:
                            # add agent's name to its messages
                            new_message["name"] = next_agent

                        self.state.replace_chat_history(
                            message_history_updated_with_agent_name
                        )  # TODO: don't hard code the messages key
                        self.state.last_agent_called = next_agent
                        span.set_outputs(
                            {
                                "post_processed_decision": next_agent,
                                "post_processing_reason": "Supervisor selected it & business logic agrees.",
                                "updated_chat_history": self.state.chat_history,
                            }
                        )

                    except ValueError as e:
                        logging.error(
                            f"Error calling agent {next_agent}: {e}.  We will default to finishing."
                        )
                        span.set_outputs(
                            {
                                "post_processed_decision": FINISH_ROUTE_NAME,
                                "post_processing_reason": "Supervisor selected an invalid agent.",
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
                        }
                    )
                    break  # finish by exiting the while loop

        # if the last message is not from the assistant, we need to add a fake assistant message
        if self.state.chat_history[-1]["role"] != "assistant":
            logging.warning(
                "No assistant ended up replying, so we'll add an error response"
            )
            self.state.append_new_message_to_history(
                {
                    "role": "assistant",
                    "content": self.agent_config.supervisor_error_response,
                }
            )

        # Return the resulting conversation back to the user
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

    input_2 = output["messages"].copy()
    input_2.append(
        {
            "role": "user",
            # "content": f"who is the user most likely to do this?",
            "content": f"how do i turn it on?",
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
