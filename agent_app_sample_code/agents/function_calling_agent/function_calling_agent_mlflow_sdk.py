# Databricks notebook source
# MAGIC %md
# MAGIC # Function Calling Agent w/ Retriever 
# MAGIC
# MAGIC In this notebook, we construct the Agent with a Retriever tool.  This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# COMMAND ----------

# # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq databricks-agents databricks-vectorsearch mlflow pydantic
# dbutils.library.restartPython()

# COMMAND ----------
import sys
# Add the parent directory to the path so we can import the `utils` modules
sys.path.append("../..")

import json
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from mlflow.deployments import get_deploy_client
import os
from utils.agents.chat import get_messages_array, extract_user_query_string, extract_chat_history
from utils.agents.config import AgentConfig, load_first_yaml_file
from utils.agents.tools import execute_function

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Agent

# COMMAND ----------

class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools
    """
    def __init__(self, agent_config: Optional[AgentConfig] = None):
        if agent_config is None:
            config_paths = [
                "../../configs/agent_model_config.yaml",
                "./configs/agent_model_config.yaml",
            ]
            self.agent_config = AgentConfig.from_yaml(load_first_yaml_file(config_paths))
        else:
            self.agent_config = agent_config
            
        self.model_serving_client = get_deploy_client("databricks")

        # Initialize the tools
        self.tool_functions = {}
        self.tool_json_schemas =[]
        for tool in self.agent_config.tools:
            self.tool_functions[tool.tool_name] = tool
            self.tool_json_schemas.append(tool.tool_input_json_schema())

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
        messages_log_with_tool_calls.append(model_response.choices[0]["message"])

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        
        return {
            "content": model_response.choices[0]["message"]["content"], #mlflow client
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.

            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        for _ in range(max_iter):
            response = self.chat_completion(messages=messages)
            assistant_message = response.choices[0]["message"]
            tool_calls = assistant_message.get('tool_calls')
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:
                function = tool_call['function']
                args = json.loads(function['arguments'])
                result = execute_function(self.tool_functions[function['name']], args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": result,
                }
                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.copy()
            del assistant_message_dict["content"]
            messages = (
                messages
                + [
                    assistant_message_dict,
                ]
                + tool_messages
            )
        raise "ERROR: max iter reached"

    def chat_completion(self, messages: List[Dict[str, str]]):
        endpoint_name = self.agent_config.llm_config.llm_endpoint_name
        llm_options = self.agent_config.llm_config.llm_parameters

        # Trace the call to Model Serving - mlflow version
        traced_create = mlflow.trace(
            self.model_serving_client.predict,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Get all tools
        tools = self.tool_json_schemas

        inputs = {
            "messages": messages,
            "tools": tools,
            **llm_options,
        }

        # Use the traced_create to make the prediction
        return traced_create(
            endpoint=endpoint_name,
            inputs=inputs,
        )


set_model(FunctionCallingAgent())
