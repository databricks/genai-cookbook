# In this file, we construct a function-calling Agent with a Retriever tool using MLflow + the OpenAI SDK connected to Databricks Model Serving. This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# Add the parent directory to the path so we can import the `cookbook` modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import json
from typing import Any, Dict, List, Optional, Union
import mlflow
import pandas as pd
from mlflow.models import set_model
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from databricks.sdk import WorkspaceClient
from cookbook.agents.utils.execute_function import execute_function

from cookbook.agents.utils.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)
from cookbook.config.agents.function_calling_agent import (
    FunctionCallingAgentConfig,
)
from cookbook.agents.utils.execute_function import execute_function
from cookbook.agents.utils.load_config import load_config
import logging

FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "function_calling_agent_config.yaml"


class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def __init__(
        self, agent_config: Optional[Union[FunctionCallingAgentConfig, str]] = None
    ):
        super().__init__()
        # Empty variables that will be initialized after loading the agent config.
        self.model_serving_client = None
        self.tool_functions = None
        self.tool_json_schemas = None
        self.chat_history = None
        self.agent_config = None

        # load the Agent's configuration. See load_config() for details.
        self.agent_config = load_config(
            passed_agent_config=agent_config,
            default_config_file_name=FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            logging.error(
                f"No agent config found.  If you are in your local development environment, make sure you either [1] are calling init(agent_config=...) with either an instance of FunctionCallingAgentConfig or the full path to a YAML config file or [2] have a YAML config file saved at {{your_project_root_folder}}/configs/{FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME}."
            )
        else:
            logging.info("Successfully loaded agent config in __init__.")

            # Now, initialize the rest of the Agent
            w = WorkspaceClient()
            self.model_serving_client = w.serving_endpoints.get_open_ai_client()

            # Initialize the tools
            self.tool_functions = {}
            self.tool_json_schemas = []
            for tool in self.agent_config.tools:
                self.tool_functions[tool.name] = tool
                self.tool_json_schemas.append(tool.get_json_schema())

            # Initialize the chat history to empty
            self.chat_history = []

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        # Check here to allow the Agent class to be initialized without a configuration file, which is required to import the class as a module in other files.
        if not self.agent_config:
            raise RuntimeError("Agent config not loaded. Cannot call predict()")

        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            # in a multi-agent setting, the last message can be from another assistant, not the user
            last_message = extract_user_query_string(messages)
            last_message_role = messages[-1]["role"]
            # Save the history inside the Agent's internal state
            self.chat_history = extract_chat_history(messages)
            span.set_outputs(
                {
                    "last_message": last_message,
                    "chat_history": self.chat_history,
                    "last_message_role": last_message_role,
                }
            )

        ##############################################################################
        # Call LLM

        # messages to send the model
        # For models with shorter context length, you will need to trim this to ensure it fits within the model's context length
        system_prompt = self.agent_config.llm_config.llm_system_prompt_template
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": last_message_role, "content": last_message}]
        )

        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        messages_log_with_tool_calls = self.recursively_call_and_run_tools(
            messages=messages
        )

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        return {
            # "content": model_response.choices[0].message.content,
            "content": messages_log_with_tool_calls[-1]["content"],
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="CHAIN")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
                response = self.chat_completion(messages=messages, tools=True)
                assistant_message = response.choices[0].message  # openai client
                tool_calls = assistant_message.tool_calls  # openai
                if tool_calls is None:
                    # the tool execution finished, and we have a generation
                    messages.append(assistant_message.to_dict())
                    return messages
                tool_messages = []
                for tool_call in tool_calls:  # TODO: should run in parallel
                    with mlflow.start_span(
                        name="execute_tool", span_type="TOOL"
                    ) as span:
                        function = tool_call.function  # openai
                        args = json.loads(function.arguments)  # openai
                        span.set_inputs(
                            {
                                "function_name": function.name,
                                "function_args_raw": function.arguments,
                                "function_args_loaded": args,
                            }
                        )
                        result = execute_function(
                            self.tool_functions[function.name], args
                        )
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }  # openai

                        tool_messages.append(tool_message)
                        span.set_outputs({"new_message": tool_message})
                assistant_message_dict = assistant_message.dict().copy()  # openai
                del assistant_message_dict["content"]
                del assistant_message_dict["function_call"]  # openai only
                if "audio" in assistant_message_dict:
                    del assistant_message_dict["audio"]  # llama70b hack
                messages = (
                    messages
                    + [
                        assistant_message_dict,
                    ]
                    + tool_messages
                )
                i += 1
        # TODO: Handle more gracefully
        raise "ERROR: max iter reached"

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.agent_config.llm_config.llm_endpoint_name
        llm_options = self.agent_config.llm_config.llm_parameters.dict()

        # # Trace the call to Model Serving - openai versio
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

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


# tell MLflow logging where to find the agent's code
set_model(FunctionCallingAgent())


# IMPORTANT: set this to False before logging the model to MLflow
debug = False

if debug:
    # logging.basicConfig(level=logging.INFO)
    # print(find_config_folder_location())
    # print(os.path.abspath(os.getcwd()))
    # mlflow.tracing.disable()
    logging.basicConfig(level=logging.DEBUG)
    agent = FunctionCallingAgent()

    vibe_check_query = {
        "messages": [
            # {"role": "user", "content": f"what is agent evaluation?"},
            {
                "role": "user",
                "content": f"How does the BlendMaster Elite 4000 blender work?",
            },
            # {
            #     "role": "user",
            #     "content": f"find all docs from the section header 'Databricks documentation archive' or 'Work with files on Databricks'",
            # },
            # {
            #     "role": "user",
            #     "content": "Translate the sku `OLD-abs-1234` to the new format",
            # }
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

    second_turn = {
        "messages": output["messages"]
        + [{"role": "user", "content": "How do I turn it on?"}]
    }

    # Run the Agent again with the same input to continue the conversation
    second_turn_output = agent.predict(model_input=second_turn)
    print(second_turn_output["content"])
