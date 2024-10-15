import json
from typing import Any, Callable, Dict, List, Optional, Union
import mlflow
from dataclasses import asdict, dataclass
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from mlflow.deployments import get_deploy_client

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
        self.config = config
        self.vector_search_client = VectorSearchClient(disable_notice=True)
        self.vector_search_index = self.vector_search_client.get_index(
            index_name=self.config.get("vector_search_index")
        )
        vector_search_schema = self.config.get("vector_search_schema")

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def get_tool_definition(self) -> Dict[str, Any]:
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
                    },
                    "required": ["query"],
                },
            },
        }

    @mlflow.trace(span_type="TOOL", name="VectorSearchRetriever")
    def __call__(self, query: str) -> str:
        results = self.similarity_search(query)

        context = ""
        for result in results:
            formatted_chunk = self.config.get("chunk_template").format(
                chunk_text=result.get("page_content"),
                metadata=json.dumps(result.get("metadata")),
            )
            context += formatted_chunk

        resulting_prompt = self.config.get("prompt_template").format(context=context)

        return resulting_prompt

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
            self.config.get("vector_search_schema").get("document_uri"),
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
        column_names = [column["name"] for column in vs_results["manifest"]["columns"]]

        docs = []
        vs_result_obj = vs_results["result"]
        if vs_result_obj["row_count"] > 0:
            for item in vs_result_obj["data_array"]:
                metadata = {}
                score = item[-1]
                if score >= vector_search_threshold:
                    metadata["similarity_score"] = score
                    chunk_text_col = self.config.get("vector_search_schema").get("chunk_text")
                    for i, column_value in enumerate(item[0:-1]):
                        column_name = column_names[i]
                        metadata[column_name] = column_value
                    # put contents of the chunk into page_content
                    page_content = metadata.pop(chunk_text_col)
                    doc = Document(page_content=page_content, metadata=metadata, type="Document")
                    docs.append(doc)

        return docs

class ToolCallingAgent(mlflow.pyfunc.PythonModel):
    """
    MLflow Pyfunc model class defining an Agent that calls tools.
    For more details on defining a custom MLflow Pyfunc model, see
    https://www.mlflow.org/docs/latest/models.html#python-function-model
    """

    def __init__(self):
        self.config = mlflow.models.ModelConfig(
            development_config="agents/generated_configs/agent.yaml"
        )

        # Load the LLM
        self.model_serving_client = get_deploy_client("databricks")

        # Init the Retriever tool
        self.retriever_tool = VectorSearchRetriever(self.config.get("retriever_tool"))

        # Configure the Review App to use the Retriever's schema
        vector_search_schema = self.config.get("retriever_tool").get(
            "vector_search_schema"
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
        messages = self._get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            user_query = self._extract_user_query_string(messages)
            # Save the history inside the Agent's internal state
            self.chat_history = self._extract_chat_history(messages)
            span.set_outputs(
                {"user_query": user_query, "chat_history": self.chat_history}
            )

        ##############################################################################
        # Generate Answer
        system_prompt = self.config.get("llm_config").get("llm_system_prompt_template")

        # Add the previous history
        messages = (
            [{"role": "system", "content": system_prompt}]
            + self.chat_history  # append chat history for multi turn
            + [{"role": "user", "content": user_query}]
        )

        # Call the LLM to call tools and eventually deliver a generation to send back to the user
        (
            model_response,
            messages_log_with_tool_calls,
        ) = self._call_llm_and_run_tools(messages=messages)

        # If your client application keeps track of conversation history and automatically
        # appends the bot's response to the passed-in messages history, remove this line.
        messages_log_with_tool_calls.append(model_response.choices[0]["message"])

        # remove the system prompt - this should not be exposed to the Agent caller
        messages_log_with_tool_calls = messages_log_with_tool_calls[1:]

        return {
            "content": model_response.choices[0]["message"]["content"],
            # this should be returned back to the Review App (or any other front end app) and stored there so it can be passed back to this stateless agent with the next turns of converastion.
            "messages": messages_log_with_tool_calls,
        }

    @mlflow.trace(span_type="AGENT")
    def _call_llm_and_run_tools(self, max_iter=10, **kwargs):
        messages = kwargs["messages"]
        del kwargs["messages"]
        for _ in range(max_iter):
            response = self._chat_completion(messages=messages, tools=True)
            assistant_message = response.choices[0]["message"]
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None:
                # the tool execution finished, and we have a generation
                return (response, messages)
            tool_messages = []
            for tool_call in tool_calls:
                function = tool_call["function"]
                args = json.loads(function["arguments"])
                result = self._execute_function(function["name"], args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                }
                tool_messages.append(tool_message)
            assistant_message_dict = assistant_message.copy()
            del assistant_message_dict["content"]
            messages.extend([assistant_message_dict] + tool_messages)
        raise "ERROR: max iter reached"

    @mlflow.trace(span_type="FUNCTION")
    def _execute_function(self, function_name, args):
        the_function = self.tool_functions.get(function_name)
        result = the_function(**args)
        return result

    def _chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.config.get("llm_config").get("llm_endpoint_name")
        llm_options = self.config.get("llm_config").get("llm_parameters")

        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.predict,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        if tools:
            # Get all tools
            tools = [self.retriever_tool.get_tool_definition()]

            inputs = {
                "messages": messages,
                "tools": tools,
                **llm_options,
            }
        else:
            inputs = {
                "messages": messages,
                **llm_options,
            }

        # Use the traced_create to make the prediction
        return traced_create(
            endpoint=endpoint_name,
            inputs=inputs,
        )

    @mlflow.trace(span_type="PARSER")
    def _get_messages_array(
        self, model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        if type(model_input) == ChatCompletionRequest:
            return model_input.messages
        elif type(model_input) == dict:
            return model_input.get("messages")
        elif type(model_input) == pd.DataFrame:
            return model_input.iloc[0].to_dict().get("messages")

    @mlflow.trace(span_type="PARSER")
    def _extract_user_query_string(
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
    def _extract_chat_history(
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
            raise ValueError(
                "chat_messages_array is not an Array of Dictionary, Pandas DataFrame, or array of MLflow Message."
            )


set_model(AgentWithRetriever())

