# Databricks notebook source
# MAGIC %md
# MAGIC # Function Calling Agent w/ Retriever
# MAGIC
# MAGIC In this notebook, we construct a function-calling Agent with a Retriever tool using MLflow + the OpenAI SDK connected to Databricks Model Serving. This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents databricks-vectorsearch "git+https://github.com/mlflow/mlflow.git" databricks-sdk[openai] pydantic "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/core" "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/integrations/openai"
# dbutils.library.restartPython()

# COMMAND ----------

import sys

# Add the parent directory to the path so we can import the `utils` modules
sys.path.append("../..")

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
from utils.agents.tools import execute_function

from utils.agents.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)
from utils.agents.function_calling_agent import FunctionCallingAgentConfig
from utils.agents.tools import execute_function
from utils.agents.utils import load_config

# COMMAND ----------


# DBTITLE 1,Agent

FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "function_calling_agent_config.yaml"


class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def load_context(self, context):
        print(context.model_config)

    def __init__(
        self, agent_config: Optional[Union[FunctionCallingAgentConfig, str]] = None
    ):
        self.agent_config = load_config(
            agent_config=agent_config,
            default_config_file_name=FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            raise ValueError("No agent config found")

        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # Initialize the tools
        self.tool_functions = {}
        self.tool_json_schemas = []
        for tool in self.agent_config.tools:
            self.tool_functions[tool.name] = tool
            self.tool_json_schemas.append(tool.get_json_schema())

        self.chat_history = []

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Call LLM

        # messages to send the model
        # For models with shorter context length, you will need to trim this to ensure it fits within the model's context length
        system_prompt = self.agent_config.llm_config.llm_system_prompt_template
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": "user", "content": user_query}]
        )

        # Call the LLM to recursively calls tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self.recursively_call_and_run_tools(messages=messages)

        # If your front end keeps of converastion history and automatically appends the bot's response to the messages history, remove this line.
        messages_log_with_tool_calls.append(
            model_response.choices[0].message.to_dict()
        )  # OpenAI client

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        return {
            "content": model_response.choices[0].message.content,
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            response = self.chat_completion(messages=messages, tools=True)
            assistant_message = response.choices[0].message  # openai client
            # assistant_message = response.choices[0]["message"] #mlflow client
            tool_calls = assistant_message.tool_calls  # openai
            # tool_calls = assistant_message.get('tool_calls')#mlflow client
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:  # TODO: should run in parallel
                function = tool_call.function  # openai
                args = json.loads(function.arguments)  # openai
                # args = json.loads(function['arguments']) #mlflow
                # result = exec_uc_func(uc_func_name, **args)
                # result = self.execute_function(function.name, args)  # openai
                result = execute_function(self.tool_functions[function.name], args)

                # result = self.execute_function(function['name'], args) #mlflow

                # format for the LLM, will throw exception if not possible
                # try:
                #     result_for_llm = json.dumps(result)
                # except Exception as e:
                #     result_for_llm = str(result)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }  # openai

                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.dict().copy()  # openai
            # assistant_message_dict = assistant_message.copy() #mlflow
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
set_model(FunctionCallingAgent)

# COMMAND ----------

# DBTITLE 1,debugging code
# IMPORTANT: set this to False before logging the model to MLflow
debug = False
# import os

# os.environ["MLFLOW_TRACKING_URI"] = "databricks"
# mlflow.set_experiment("/Users/eric.peter@databricks.com/my_agent_mlflow_experiment")

if debug:
    agent = FunctionCallingAgent()

    vibe_check_query = {
        "messages": [
            # {"role": "user", "content": f"what is agent evaluation?"},
            {
                "role": "user",
                "content": f"find all docs from the section header 'Databricks documentation archive' or 'Work with files on Databricks'",
            },
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
    print(output)
