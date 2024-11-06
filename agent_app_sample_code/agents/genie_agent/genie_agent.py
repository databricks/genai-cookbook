# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Space Agent
# MAGIC
# MAGIC In this notebook, we construct a Genie space as an Agent. This Agent is encapsulated in a MLflow PyFunc class called `GenieAgent()`.

# COMMAND ----------

# # # If running this notebook by itself, uncomment these.
# %pip install --upgrade -qqqq mlflow databricks-sdk tabulate tiktoken
# dbutils.library.restartPython()

# COMMAND ----------

import sys

# Add the parent directory to the path so we can import the `utils` modules
sys.path.append("../..")

import json
from typing import Any, Dict, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
import tiktoken
import logging
import uuid
import time
from datetime import datetime
from typing import Union
import pandas as pd
from utils.agents.genie_agent import GenieAgentConfig
from utils.agents.yaml_loader import load_first_yaml_file
from utils.agents.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
    convert_messages_to_open_ai_format,
    concat_messages_array_to_string,
)
from databricks.sdk import WorkspaceClient

logging.getLogger().setLevel(logging.INFO)


# COMMAND ----------

CONFIG_FILE_NAME = "genie_config.yaml"

# COMMAND ----------

MAX_TOKENS_OF_DATA = 20000  # max tokens of data in markdown format
MAX_ITERATIONS = 50  # max times to poll the API when polling for either result or the query results, each iteration is ~1 second, so max latency == 2 * MAX_ITERATIONS


@mlflow.trace(span_type="PARSER")
def _parse_query_result(resp) -> Union[str, pd.DataFrame]:
    columns = resp["manifest"]["schema"]["columns"]
    header = [str(col["name"]) for col in columns]
    rows = []
    output = resp["result"]
    if not output:
        return None

    for item in resp["result"]["data_typed_array"]:
        row = []
        for column, value in zip(columns, item["values"]):
            type_name = column["type_name"]
            str_value = value.get("str", None)
            if str_value is None:
                row.append(None)
                continue

            if type_name in ["INT", "LONG", "SHORT", "BYTE"]:
                row.append(int(str_value))
            elif type_name in ["FLOAT", "DOUBLE", "DECIMAL"]:
                row.append(float(str_value))
            elif type_name == "BOOLEAN":
                row.append(str_value.lower() == "true")
            elif type_name == "DATE":
                row.append(datetime.strptime(str_value[:10], "%Y-%m-%d").date())
            elif type_name == "TIMESTAMP":
                row.append(datetime.strptime(str_value[:10], "%Y-%m-%d").date())
            elif type_name == "BINARY":
                row.append(bytes(str_value, "utf-8"))
            else:
                row.append(str_value)

        rows.append(row)

    # initial parsing to markdown
    query_result = pd.DataFrame(rows, columns=header).to_markdown()

    # trim down from the total rows
    trimmed_rows = len(rows)
    tokens_used = count_tokens(query_result)

    # if the first iteration is < MAX_TOKENS_OF_DATA it will just return and skip this loop
    while trimmed_rows > 0 and tokens_used > MAX_TOKENS_OF_DATA:
        with mlflow.start_span(name="reduce_data_tokens") as span:
            span.set_inputs(
                {
                    "output_rows_to_show": trimmed_rows,
                    "max_tokens_target": MAX_TOKENS_OF_DATA,
                }
            )
            # convert to markdown
            query_result = (
                pd.DataFrame(rows, columns=header).head(trimmed_rows).to_markdown()
            )
            # keep trimming down until we get under the token limit
            trimmed_rows -= 5
            # worst case, return None, which the Agent will handle and not display the query results
            tokens_used = count_tokens(query_result)
            if trimmed_rows == 0:
                query_result = None
                tokens_used = 0
            span.set_outputs({"query_result": query_result, "tokens_used": tokens_used})
    return query_result.strip() if query_result else query_result


# Define a function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))


@dataclass
class GenieResponse:
    sql_query: str = None  # generated sql query
    response: str = (
        None  # description of the sql query or Genie's response back to the user
    )
    data_table: str = None  # datatable returned formatted as markdown by pandas


class GenieAPIWrapper:
    def __init__(
        self,
        space_id,
        encountered_error_user_message: str = "I encountered an error trying to answer your question, please try again.",
    ):
        self.space_id = space_id

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        workspace_client = WorkspaceClient()
        self._genie_client = workspace_client.genie
        self.encountered_error_user_message = encountered_error_user_message

        # We build the GenieResponse throughout this wrapper's logic since you must poll for the result & the results come back from multiple polling requests.
        self.genie_result = GenieResponse()

    @mlflow.trace()
    def start_conversation(self, content):
        resp = self._genie_client._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/start-conversation",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def create_message(self, conversation_id, content):
        resp = self._genie_client._api.do(
            "POST",
            f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages",
            body={"content": content},
            headers=self.headers,
        )
        return resp

    @mlflow.trace()
    def poll_for_result(self, conversation_id, message_id):
        @mlflow.trace()
        def poll_result():
            iteration_count = 0
            while True and iteration_count < MAX_ITERATIONS:
                # try:  # genie API randomly crashes with BadRequest: Message <id> does not have a query statementId.  This is instead caught in the Agent itself to capture all unknown exceptions from the API wrapper.
                iteration_count += 1
                logging.debug(
                    f"Polling for result {message_id} {conversation_id} iteration {iteration_count}"
                )
                resp = self._genie_client._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}",
                    headers=self.headers,
                )
                logging.debug(f"Genie polling response: {resp}")
                if resp["status"] == "EXECUTING_QUERY":
                    with mlflow.start_span(name="get_sql_query") as span:
                        query_result = next(
                            r for r in resp["attachments"] if "query" in r
                        )["query"]
                        span.set_inputs(resp)
                        self.genie_result.sql_query = query_result.get("query")
                        self.genie_result.response = query_result.get("description")
                        span.set_outputs(
                            {
                                "sql_query": self.genie_result.sql_query,
                                "response": self.genie_result.response,
                            }
                        )
                    return poll_query_results()
                elif resp["status"] == "COMPLETED":
                    """
                    Genie didn't run a query, returned a question or comment to the user
                    """
                    with mlflow.start_span(name="get_genie_response") as span:
                        logging.debug(f"Genie polling returned {resp}")
                        span.set_inputs(resp)
                        # Get first attachment from array safely
                        first_attachment = (
                            resp.get("attachments", [])[0]
                            if resp.get("attachments")
                            else None
                        )
                        if first_attachment:
                            # TODO: we shouldn't need this logic, but it's here to handle a bug in the Genie API where sometimes you get COMPLETED before EXECUTING_QUERY is returned.
                            if "text" in first_attachment:
                                # genie didn't run a query, just returned a question or comment to the user
                                response = first_attachment["text"]["content"]
                                self.genie_result.response = response
                                span.set_outputs(
                                    {"response": self.genie_result.response}
                                )
                                return asdict(self.genie_result)
                            elif "query" in first_attachment:
                                # genie ran a query, get the results
                                response = first_attachment["query"]["description"]
                                self.genie_result.sql_query = first_attachment["query"][
                                    "query"
                                ]
                                self.genie_result.response = first_attachment["query"][
                                    "description"
                                ]
                                span.set_outputs(
                                    {
                                        "sql_query": self.genie_result.sql_query,
                                        "response": self.genie_result.response,
                                    }
                                )
                                return poll_query_results()
                            else:
                                # unknown state, assume an error state
                                self.genie_result.response = (
                                    self.encountered_error_user_message
                                )
                                span.set_outputs(
                                    {"response": self.genie_result.response}
                                )
                                return asdict(self.genie_result)
                        else:
                            # no response, must be an error state
                            self.genie_result.response = (
                                self.encountered_error_user_message
                            )
                            span.set_outputs({"response": self.genie_result.response})
                            return asdict(self.genie_result)

                elif resp["status"] == "FAILED":
                    """
                    Genie failed
                    """
                    self.genie_result.response = self.encountered_error_user_message
                    return asdict(self.genie_result)
                else:
                    logging.debug(f"Waiting...: {resp['status']}")
                    time.sleep(1)
                # except Exception as e:  # hack per above
                #     logging.error(
                #         f"Error polling for result: {e}, in polling iteration {iteration_count} of {MAX_ITERATIONS}"
                #     )
                #     print(iteration_count)
                #     continue

        @mlflow.trace()
        def poll_query_results():
            iteration_count = 0
            while True and iteration_count < MAX_ITERATIONS:
                iteration_count += 1
                resp = self._genie_client._api.do(
                    "GET",
                    f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/query-result",
                    headers=self.headers,
                )["statement_response"]

                state = resp["status"]["state"]
                if state == "SUCCEEDED":
                    with mlflow.start_span(name="get_sql_query_results") as span:
                        span.set_inputs(resp)
                        data_table_as_md = _parse_query_result(resp)
                        self.genie_result.data_table = data_table_as_md
                        span.set_outputs(self.genie_result.data_table)
                    return asdict(self.genie_result)
                elif state == "RUNNING" or state == "PENDING":
                    logging.debug("Waiting for query result...")
                    time.sleep(1)
                else:
                    logging.debug(f"No query result: {resp['state']}")
                    return None

        return poll_result()

    @mlflow.trace(span_type="AGENT", name="genie")
    def ask_question(self, question):
        self.genie_result = GenieResponse()
        resp = self.start_conversation(question)
        return self.poll_for_result(resp["conversation_id"], resp["message_id"])


# COMMAND ----------


# DBTITLE 1,Agent
class GenieAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that does function-calling with tools using OpenAI SDK
    """

    def load_context(self, context):
        print("load_context")
        print(context.model_config)

    def __init__(self, agent_config: Optional[GenieAgentConfig] = None):
        print("init")
        if agent_config is None:
            # Try to load from local config file first for inner dev loop; in serving env these files will not be present, so load the model's logged config e.g., the config from mlflow.pyfunc.log_model(model_config=...) via mlflow.ModelConfig()
            config_paths = [
                f"../../configs/{CONFIG_FILE_NAME}",
                f"./configs/{CONFIG_FILE_NAME}",
            ]
            try:
                self.agent_config = GenieAgentConfig.from_yaml(
                    load_first_yaml_file(config_paths)
                )
            except ValueError as e:
                logging.info(
                    f"No local config YAML found at {config_paths}, loading mlflow.ModelConfig() instead."
                )
                model_config = mlflow.models.ModelConfig()
                self.agent_config = GenieAgentConfig(**model_config.to_dict())
                logging.info(
                    f"Loaded GenieAgentConfig from mlflow.ModelConfig(): {self.agent_config}"
                )
        else:
            self.agent_config = agent_config

        self.test = mlflow.models.ModelConfig(
            # development_config=f"../../configs/{CONFIG_FILE_NAME}"
        )
        print(self.test.get("genie_space_id"))

        # Load the API wrapper
        self._genie_agent = GenieAPIWrapper(self.agent_config.genie_space_id)

        self.chat_history = []

    @mlflow.trace(name="genie_orchestator", span_type="AGENT")
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

        # HACK: Since Genie API doesn't provide a stateless API that you can pass the chat history in, we "prompt hack" Genie by adding the chat history to the user's query.
        # This avoids the need for this agent to maintain the genie converastion ID between turns - which is impracticle since this Agent is deployed as a stateless API.
        if len(self.chat_history) > 0:
            message = f"I will provide you a chat, where your name is 'assistant' and the user is 'user'. Please help with the user's last query.  DO NOT reference the query or context in your response.\n"

            # Concatenate messages to form the chat history
            message += concat_messages_array_to_string(messages)
        else:
            message = user_query

        # Send the message and wait for a response
        try:
            genie_response = self._genie_agent.ask_question(message)
        except (
            Exception
        ) as e:  # genie API randomly crashes with BadRequest: Message <id> does not have a query statementId
            genie_response = None
            logging.error(f"Error calling Genie API wrapper: {e}.")

        if genie_response:
            if genie_response["data_table"]:
                output_message = (
                    f"{genie_response['response']}\n\n{genie_response['data_table']}"
                )
            else:
                output_message = f"{genie_response['response']}"
        else:
            output_message = self.agent_config.encountered_error_user_message

        with mlflow.start_span(name="update_message_history") as span:
            # message log
            # only put the actual query in it
            message_log = convert_messages_to_open_ai_format(messages)
            # add a fake tool call version of genie so we can debug this in the MLflow UIs
            message_log += self.get_faked_tool_calls(user_query, genie_response)
            # add genie's text response
            message_log.append({"role": "assistant", "content": output_message})
            span.set_outputs(message_log)

        return {
            "content": output_message,
            # messages should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": message_log,
        }

    @mlflow.trace()
    def get_faked_tool_calls(self, user_query, genie_response):
        fake_id = str(uuid.uuid4().hex)
        args = {"query": user_query}
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{fake_id}",
                        "function": {"arguments": json.dumps(args), "name": "genie"},
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call_{fake_id}",
                "content": json.dumps(genie_response),
            },
        ]


# tell MLflow logging where to find the agent's code
set_model(GenieAgent())

# COMMAND ----------

debug = False
if debug:
    # mlflow.tracing.disable()
    agent = GenieAgent()

    vibe_check_query = {
        "messages": [
            {
                "role": "user",
                # "content": f"What is the churn rate?",
                # "content": f"a irrelevant question",
                "content": f"what tables you got?",
            },
        ]
    }

    output = agent.predict(model_input=vibe_check_query)
    print(output)
