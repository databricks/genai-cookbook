# In this file, we construct a function-calling Agent with a Retriever tool using MLflow + the OpenAI SDK connected to Databricks Model Serving. This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# Add the parent directory to the path so we can import the `cookbook` modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import os
import json
from typing import Any, Dict, List, Optional, Union
import mlflow
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from databricks.sdk import WorkspaceClient
from autogen import ConversableAgent
from autogen import register_function

from cookbook.agents.utils.execute_function import execute_function
from cookbook.agents.utils.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)
from cookbook.config.agents.function_calling_agent import (
    FunctionCallingAgentConfig,
)
from cookbook.tools.uc_tool import UCTool
from cookbook.agents.utils.execute_function import execute_function
from cookbook.agents.utils.load_config import load_config
from cookbook.agents.utils.databricks_model_serving_client import DatabricksModelServingClient
import logging


FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "function_calling_agent_config.yaml"

class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using Autogen
    """

    def __init__(
        self,
        agent_config: Optional[Union[FunctionCallingAgentConfig, str]] = None
    ):
        super().__init__()
        # Empty variables that will be initialized after loading the agent config.
        self.agent_config = None
        self.tools = None

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
            self.tools = self.agent_config.tools

    def create_agents(self, chat_history):

        def is_termination_message(message):
            content = message.get("content", "")
            return (content and "TERMINATE" in content.upper()) or (message['role'] == 'user' and 'tool_calls' not in message)

        # The user proxy agent is used for interacting with the assistant agent
        # and executes tool calls.
        user_proxy = ConversableAgent(
            name="User",
            llm_config=False,
            is_termination_msg=is_termination_message,
            human_input_mode="NEVER",
        )
        
        llm_config = self.agent_config.llm_config
        
        system_prompt = llm_config.llm_system_prompt_template

        config_list = [{
                        "model_client_cls": "DatabricksModelServingClient",
                        "model": llm_config.llm_endpoint_name,
                        "endpoint_name": llm_config.llm_endpoint_name,
                        "llm_config": llm_config.llm_parameters.dict()}]

        assistant = ConversableAgent(
            name="Assistant",
            system_message=system_prompt,
            llm_config={"config_list": config_list, "cache_seed": None},
            chat_messages={user_proxy: chat_history}
        )

        for tool in self.tools:
            if isinstance(tool, UCTool):
                tool._toolkit.tools[0].register_function(callers = assistant,
                                executors = user_proxy )
            else:
                register_function(tool,
                                    caller=assistant,
                                    executor=user_proxy,  
                                    name=tool.name,
                                    description=tool.description)

        return assistant, user_proxy
    

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
            last_message_content = extract_user_query_string(messages)
            last_message_role = messages[-1]["role"]
            last_message = {"role": last_message_role, "content": last_message_content}
            # Save the history inside the Agent's internal state
            chat_history = extract_chat_history(messages)
            span.set_outputs(
                {
                    "last_message": last_message,
                    "chat_history": chat_history
                }
            )

        ##############################################################################
        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self.recursively_call_and_run_tools(last_message=last_message, 
                                                chat_history=chat_history)

        return {
            "content": model_response['content'],
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, 
                                       last_message, 
                                       chat_history,
                                       last_max_iter=10):

        assistant, user_proxy = self.create_agents(chat_history)

        assistant.register_model_client(model_client_cls=DatabricksModelServingClient)

        model_response = user_proxy.initiate_chat(assistant, 
                                                  message=last_message['content'],
                                                  max_turns=last_max_iter,
                                                  clear_history=False)

        return assistant.last_message(user_proxy), assistant.chat_messages[user_proxy]


logging.basicConfig(level=logging.INFO)

# tell MLflow logging where to find the agent's code
set_model(FunctionCallingAgent())


# IMPORTANT: set this to False before logging the model to MLflow
debug = False

if debug:
    # logging.basicConfig(level=logging.INFO)
    # print(find_config_folder_location())
    # print(os.path.abspath(os.getcwd()))
    # mlflow.tracing.disable()
    agent = FunctionCallingAgent()

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
