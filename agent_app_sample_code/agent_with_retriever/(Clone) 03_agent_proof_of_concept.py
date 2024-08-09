# Databricks notebook source
# MAGIC %md
# MAGIC # Agent proof of concept.
# MAGIC
# MAGIC By the end of this notebook, you will have created a POC of your Agent that you can interact with, and ask questions.
# MAGIC
# MAGIC This means:
# MAGIC - We will have a mlflow model registered in the "Models" tab on the Databricks menu on the left. Models that are registered are just assets that can be instantiated from another notebook, but are not served on an endpoint. These models can be invoked with `mlflow.invoke()`.
# MAGIC - We will have a served model registered in the "Serving" tab on the Databricks menu on the left. This means that the model is served and can be accessed via a UI or a REST API for anyone in the workspace.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade -qqqq databricks-agents openai databricks-vectorsearch databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the global configuration

# COMMAND ----------

# MAGIC %run ./00_global_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration

# COMMAND ----------

# MAGIC %run ./agents/agent_config

# COMMAND ----------

# If you used the 02_data_pipeline notebook, you can use this cell to get the configuration of the Retriever

# TODO: Add this support

# COMMAND ----------

# TODO: This must be manually set by the user.  See TODO above for providing a way to automatically set this from 02_data_pipeline
retriever_config = RetrieverToolConfig(
    vector_search_index="ericpeter_catalog.agents.db_docs_app_chunked_docs_index",
    vector_search_schema=RetrieverSchemaConfig(
        primary_key="chunk_id",
        chunk_text="content_chunked",
        document_uri="doc_uri",
        additional_metadata_columns=[],
    ),
    parameters=RetrieverParametersConfig(num_results=5, query_type="ann"),
    vector_search_threshold=0.1,
    chunk_template="Passage text: {chunk_text}\nPassage metadata: {metadata}\n\n",
    prompt_template="""Use the following pieces of retrieved context to answer the question.\nOnly use the passages from context that are relevant to the query to answer the question, ignore the irrelevant passages.  When responding, cite your source, referring to the passage by the columns in the passage's metadata.\n\nContext: {context}""",
    tool_description_prompt="Search for documents that are relevant to a user's query about the [REPLACE WITH DESCRIPTION OF YOUR DOCS].",
)

# TODO: Improve these docs
# `llm_endpoint_name`: Model Serving endpoint with the LLM for your Agent. 
#     - Either an [Foundational Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) Provisioned Throughput / Pay-Per-Token or [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) of type `/llm/v1/chat` with support for [function calling](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html).  Supported models: `databricks-meta-llama-3-70b-instruct` or any of the Azure OpenAI / OpenAI models.

llm_config = LLMConfig(
    # https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    # llm_endpoint_name="databricks-meta-llama-3-70b-instruct",
    llm_endpoint_name="ep-gpt4o",
    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template=(
        """You are a helpful assistant that answers questions by calling tools.  Provide responses ONLY based on the information from tools that are explictly specified to you.  If you do not have a relevant tool for a question, respond with 'Sorry, I'm not trained to answer that question'."""
    ),
    # Parameters that control how the LLM responds.
    llm_parameters=LLMParametersConfig(temperature=0.01, max_tokens=1500),
)

agent_config = AgentConfig(
    # TODO: Make this generalized to include multiple tools
    retriever_tool=retriever_config,
    llm_config=llm_config,
    input_example={
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
)

agent_config.dict()

# COMMAND ----------

# MAGIC %run ./validators/validate_agent_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set the MLflow experiement name
# MAGIC
# MAGIC Used to store the Agent's model

# COMMAND ----------

import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# COMMAND ----------

# Use OpenAI client with Model Serving
# TODO: Improve the docs for why this happens
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN
os.environ["DATABRICKS_HOST"] = f"{API_ROOT}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the Agent's code

# COMMAND ----------

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from openai import OpenAI
from databricks.vector_search.client import VectorSearchClient

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, str]
    type: str


class VectorSearchRetriever:
    """
    Class using Databricks Vector Search to retrieve relevant documents.
    """

    def __init__(self, config: Dict[str, Any]):
        # TODO: Validate config 
        self.config = config
        self.vector_search_client = VectorSearchClient(disable_notice=True)
        self.vector_search_index = self.vector_search_client.get_index(
            index_name=self.config.get("vector_search_index")
        )
        vector_search_schema = self.config.get("vector_search_schema")

    def get_config(self) -> Dict[str, Any]:
        return self.config
    
    def get_tool_definition(self) -> Dict[str, Any]:
        # description = "Search for documents that are relevant to a user's query."
        # if self.config.get("description_prompt"):
        #       description = description + f"  This tool contains documents about {self.config.get('description_prompt')}"

        return {
            "type": "function",
            "function": {
                "name": "retrieve_documents",
                "description": self.config.get("tool_description_prompt"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to find documents about.",
                        },
                        # "doc_name_filter": {
                        #     "type": "string",
                        #     "enum": [
                        #         "/Volumes/ericpeter_catalog/agents/source_docs/2212.14024.pdf",
                        #         "test_doc_2",
                        #     ],
                        #     "description": "A filter for the specific document name.",
                        # },
                    },
                    "required": ["query"],
                },
            },
        }

    @mlflow.trace(span_type="TOOL", name="VectorSearchRetriever")
    def __call__(self, query: str) -> str:
        # TODO: Rewrite the query e.g., "what is it?" to "what is [topic from previous question]".  Test the chain with and without this - some function calling models automatically handle the query rewriting e.g., when they call the tool, they rewrite the query

        # print(doc_name_filter)
        results = self.similarity_search(query)

        context = ""
        for result in results:
            formatted_chunk = self.config.get("chunk_template").format(chunk_text=result.get("page_content"), metadata=json.dumps(result.get("metadata")))
            context += formatted_chunk 

        resulting_prompt = (
            self.config.get("prompt_template")
            .format(context=context)
        )

        return resulting_prompt  # json.dumps(results, default=str)

    @mlflow.trace(span_type="RETRIEVER")
    def similarity_search(
        self, query: str, filters: Dict[Any, Any] = None
    ) -> List[Document]:
        """
        Performs vector search to retrieve relevant chunks.

        Args:
            query: Search query.
            filters: Optional filters to apply to the search, must follow the Databricks Vector Search filter spec (https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#use-filters-on-queries)

        Returns:
            List of retrieved Documents.
        """

        traced_search = mlflow.trace(
            self.vector_search_index.similarity_search,
            name="vector_search.similarity_search",
        )

        # print(self.config)
        columns = [
            self.config.get("vector_search_schema").get("primary_key"),
            self.config.get("vector_search_schema").get("chunk_text"),
            self.config.get("vector_search_schema").get("document_uri")
        ] + self.config.get("vector_search_schema").get("additional_metadata_columns")

        if filters is None:
            results = traced_search(
                query_text=query,
                columns=columns,
                **self.config.get("parameters"),
            )
        else:
            results = traced_search(
                query_text=query,
                filters=filters,
                columns=columns,
                **self.config.get("parameters"),
            )

        vector_search_threshold = self.config.get("vector_search_threshold")
        documents = self.convert_vector_search_to_documents(
            results, vector_search_threshold
        )

        return [asdict(doc) for doc in documents]

    @mlflow.trace(span_type="PARSER")
    def convert_vector_search_to_documents(
        self, vs_results, vector_search_threshold
    ) -> List[Document]:
        column_names = []
        for column in vs_results["manifest"]["columns"]:
            column_names.append(column)

        docs = []
        if vs_results["result"]["row_count"] > 0:
            for item in vs_results["result"]["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata["similarity_score"] = score
                    # print(score)
                    i = 0
                    for field in item[0:-1]:
                        # print(field + "--")
                        metadata[column_names[i]["name"]] = field
                        i = i + 1
                    # put contents of the chunk into page_content
                    page_content = metadata[
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]
                    del metadata[
                        self.config.get("vector_search_schema").get("chunk_text")
                    ]

                    doc = Document(
                        page_content=page_content, metadata=metadata, type="Document"
                    )  # , 9)
                    docs.append(doc)

        return docs
      
class AgentWithRetriever(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that includes a Retriever tool
    """
    def __init__(self, agent_config: dict = None):
        self.__agent_config = agent_config
        if self.__agent_config is None:
            self.__agent_config = globals().get("__mlflow_model_config__")

        print(globals().get("__mlflow_model_config__"))

    def load_context(self, context):
        # TODO: This is an ugly hack to support the CUJ of iterating on config in a notebook
        print(globals().get("__mlflow_model_config__"))
        if self.__agent_config is None:
            try:
                self.config = mlflow.models.ModelConfig(development_config="agents/generated_configs/agent.yaml")
            except Exception as e:
                self.config = mlflow.models.ModelConfig()
        else:
            self.config = mlflow.models.ModelConfig(development_config=self.__agent_config)
        
        # Load the LLM
        # OpenAI client used to query Databricks Model Serving
        self.model_serving_client = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN"),
            base_url=str(os.environ.get("DATABRICKS_HOST")) + "/serving-endpoints",
        )

        # print(self.config)

        # Init the Retriever tool
        self.retriever_tool = VectorSearchRetriever(
            self.config.get("retriever_tool")
        )

        # Configure the Review App to use the Retriever's schema
        vector_search_schema = (
            self.config.get("retriever_tool")
            .get("vector_search_schema")
        )
        mlflow.models.set_retriever_schema(
            primary_key=vector_search_schema.get("primary_key"),
            text_column=vector_search_schema.get("chunk_text"),
            doc_uri=vector_search_schema.get("doc_uri"),
        )

        self.tool_functions = {
            "retrieve_documents": self.retriever_tool,
        }

        # Internal representation of the chat history.  As the Agent iteratively selects/executes tools, the history will be stored here.  Since the Agent is stateless, this variable must be populated on each invocation of `predict(...)`.
        self.chat_history = None

    @mlflow.trace(name="chain", span_type="CHAIN")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = self.get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = self.extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = self.extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Generate Answer
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")

        # Add the previous history
        # TODO: Need a way to include the previous tool calls
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
        messages_log_with_tool_calls.append(model_response.choices[0].message.to_dict())

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        return {
            "content": model_response.choices[0].message.content,
            # TODO: this should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="TOOL")
    def retrieve_documents(self, query, doc_name_filter) -> Dict:
        # Rewrite the query e.g., "what is it?" to "what is [topic from previous question]".  Test the chain with and without this - some function calling models automatically handle the query rewriting e.g., when they call the tool, they rewrite the query
        vs_query = query
        # if len(self.chat_history) > 0:
        #     vs_query = self.query_rewrite(query, self.chat_history)
        # else:
        #     vs_query = query

        print(doc_name_filter)
        results = self.customer_notes_retriever.similarity_search(vs_query)

        context = ""
        for result in results:
            context += "Document: " + json.dumps(result) + "\n"

        resulting_prompt = (
            self.config.get("retriever_tool")
            .get("prompt_template")
            .format(context=context)
        )

        return resulting_prompt  # json.dumps(results, default=str)

    @mlflow.trace(span_type="PARSER")
    def query_rewrite(self, query, chat_history) -> str:
        ############
        # Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
        ############
        query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

        Chat history: {chat_history}

        Question: {question}"""

        chat_history_formatted = self.format_chat_history(chat_history)

        prompt = query_rewrite_template.format(
            question=query, chat_history=chat_history_formatted
        )

        model_response = self.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        if len(model_response.choices) > 0:
            return model_response.choices[0].message.content
        else:
            # if no generation, return the original query
            return query

    @mlflow.trace(span_type="AGENT")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        # tools = self.config.get("tools")
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            response = self.chat_completion(messages=messages, tools=True)
            # response = client.chat.completions.create(tools=tools, messages=messages, **kwargs)
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                # print(response)
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:  # TODO: should run in parallel
                function = tool_call.function
                # uc_func_name = decode_function_name(function.name)
                args = json.loads(function.arguments)
                # result = exec_uc_func(uc_func_name, **args)
                result = self.execute_function(function.name, args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.dict()
            del assistant_message_dict["content"]
            del assistant_message_dict["function_call"]
            messages = (
                messages
                + [
                    assistant_message_dict,
                ]
                + tool_messages
            )
            # print("---current state of messages---")
            # print(messages)
        raise "ERROR: max iter reached"

    @mlflow.trace(span_type="FUNCTION")
    def execute_function(self, function_name, args):
        the_function = self.tool_functions.get(function_name)
        # print(the_function)
        result = the_function(**args)
        return result

    # @mlflow.trace(span_type="CHAT_MODEL", name="chat_completions_wrapper")
    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        if tools:
            # Get all tools
            # TODO: Generalize this to work with tools other than a hard-coded Retriever tool
            tools = [self.retriever_tool.get_tool_definition()]

            return traced_create(
                model=endpoint_name,
                messages=messages,
                tools=tools,
                **llm_options,
            )
        else:
            # Call LLM without any tools
            return traced_create(model=endpoint_name, messages=messages, **llm_options)

    @mlflow.trace(span_type="PARSER")
    def get_messages_array(
        self, model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        # TODO: This is required to handle both Dict + ChatCompletionRequest wrapped inputs.  If ChatCompletionRequest  supported .get(), this code wouldn't be required.

        if type(model_input) == ChatCompletionRequest:
            return model_input.messages
        elif type(model_input) == dict:
            return model_input.get("messages")
        ## required to test with the following code after logging
        ## model = mlflow.pyfunc.load_model(model_info.model_uri)
        ## model.predict(agent_config['input_example'])
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
            chat_messages_array: Array of Dictionary representing each chat messages.

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
            raise ValueError("chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message.")

    @mlflow.trace(span_type="PARSER")
    def format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Formats the chat history into a string.

        Args:
            chat_history: List of chat messages.

        Returns:
            Formatted chat history string.
        """
        if not chat_history:
            return ""

        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append(f"User: {message['content']}")

            # this logic ignores assistant messages that are just about tool calling and have no user facing content
            elif message["role"] == "assistant" and message.get("content"):
                formatted_history.append(f"Assistant: {message['content']}")

        return "\n".join(formatted_history)


set_model(AgentWithRetriever())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Iteratively vibe check & adjust the Agent's configuration
# MAGIC
# MAGIC If you need to adjust the Agent's code, you can do so in the `./agents/function_calling_agent_w_retriever_tool` Notebook

# COMMAND ----------

# Load the Agent
agent = AgentWithRetriever(agent_config=agent_config.dict())
agent.load_context(None)

# Example for testing multiple turns of converastion

# TODO: Show how to run Agent Evaluation here to help with Vibe checking

# 1st turn of converastion
first_turn_input = {
    "messages": [
        {"role": "user", "content": f"what is lakehouse monitoring?"},
    ]
}

response = agent.predict(model_input=first_turn_input)
print(response["content"])

print()
print("------")
print()

# 2nd turn of converastion
new_messages = response["messages"]
new_messages.append({"role": "user", "content": f"how do i use it?"})
# print(type(new_messages))
second_turn_input = {"messages": new_messages}
response = agent.predict(model_input=second_turn_input)
print(response["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC * Logging the model
# MAGIC * Evaluating the model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Agent POC to the Review App

# COMMAND ----------

# TODO: Remove the need for this w/ automatic credential support

w = WorkspaceClient()

# Where to save the secret
SCOPE_NAME = "ep"
SECRET_NAME = "llm_chain_pat_token"

# PAT token
SECRET_TO_SAVE = "" # REMOVED

existing_scopes = [scope.name for scope in w.secrets.list_scopes()]
if SCOPE_NAME not in existing_scopes:
    print(f"Creating secret scope `{SCOPE_NAME}`")
    w.secrets.create_scope(scope=SCOPE_NAME)
else:
    print(f"Secret scope `{SCOPE_NAME}` exists")

existing_secrets = [secret.key for secret in w.secrets.list_secrets(scope=SCOPE_NAME)]
if SECRET_NAME not in existing_secrets:
    print(f"Saving secret to `{SCOPE_NAME}.{SECRET_NAME}`")
    w.secrets.put_secret(scope=SCOPE_NAME, key=SECRET_NAME, string_value=SECRET_TO_SAVE)
else:
    print(f"Secret named `{SCOPE_NAME}.{SECRET_NAME}` already exists")


# COMMAND ----------

import pkg_resources

def get_package_version(package_name):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        return package_version
    except pkg_resources.DistributionNotFound:
        return f"{package_name} is not installed"

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
import yaml

databricks_resources = [
    DatabricksServingEndpoint(endpoint_name=llm_config.llm_endpoint_name),
    # TODO: Add the embedding model here
    DatabricksVectorSearchIndex(index_name=retriever_config.vector_search_index)
]

# Specify the full path to the Agent notebook 
model_file = "agents/function_calling_agent_w_retriever_tool"
model_path = os.path.join(os.getcwd(), model_file)

# Dump the config so the agent can use it for testing locally
# TODO: This should be automatically handled by MLflow - for some reason, MLflow doesn't inject the logged configuraiton into the model when loading the model locally with mlflow.pyfunc.load_model(model_info.model_uri)
chain_config_filepath = 'agents/generated_configs/agent.yaml'
with open(chain_config_filepath, 'w') as f:
  yaml.dump(agent_config.dict(), f)

__mlflow_model_config__ = agent_config.dict()

with mlflow.start_run(run_name=POC_CHAIN_RUN_NAME):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_path,
        # model_config=agent_config.dict(), # DOES NOT WORK
        model_config = os.path.join(os.getcwd(), chain_config_filepath),  # DOES NOT WORK EITHER
        artifact_path="agent",
        input_example=agent_config.input_example,
        resources=databricks_resources,
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        signature=ModelSignature(
            inputs=ChatCompletionRequest(),
            outputs=StringResponse(),
        ),
        # specify all python packages that are required by your Agent
        pip_requirements=[
            "openai==" + get_package_version("openai"),
            "databricks-agents==" + get_package_version("databricks-agents"),
            "databricks-vectorsearch=="+get_package_version("databricks-vectorsearch"),
        ],
    )

# COMMAND ----------

### Test the logged model
model = mlflow.pyfunc.load_model(model_info.model_uri)
model.predict(agent_config.input_example)

# COMMAND ----------

# Use Unity Catalog as the model registry
mlflow.set_registry_uri('databricks-uc')

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=model_info.model_uri, 
                                                 name=UC_MODEL_NAME)

# COMMAND ----------

from databricks import agents
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate


# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(UC_MODEL_NAME, 
                                uc_registered_model_info.version,
                                environment_vars={"DATABRICKS_HOST" : 'https://' + mlflow.utils.databricks_utils.get_workspace_url(), 
                                                  "DATABRICKS_TOKEN": "{{secrets/"+SCOPE_NAME+"/"+SECRET_NAME+"}}"}
                                )

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")
