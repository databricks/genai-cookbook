from databricks.sdk import WorkspaceClient
from openai import OpenAI
import pandas as pd
from typing import Any, Union, Dict, List
import mlflow
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
import dataclasses
import json

DEFAULT_CONFIG = {
    'endpoint_name': "agents-demo-gpt4o",
    'temperature': 0.01,
    'max_tokens': 1000,
    'system_prompt': """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
    'max_context_chars': 4096 * 4
}

RETRIEVER_TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "search_product_docs",
        "description": "Use this tool to search for Databricks product documentation.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "additionalProperties": False,
            "properties": {
                "query": {
                    "description": "a set of individual keywords to find relevant docs for.  each item of the array must be a single word.",
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
        },
    },
}]

class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing a function-calling Agent that has one tool: a retriever w/ keyword-based search.
    """

    def __init__(self):
        """
        Initialize the OpenAI SDK client connected to Model Serving.
        Load the Agent's configuration from MLflow Model Config.
        """
        # Initialize OpenAI SDK connected to Model Serving
        w = WorkspaceClient()
        self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()

        # Load config
        self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)

        # Configure playground & review app & agent evaluation to display / see the chunks from the retriever 
        mlflow.models.set_retriever_schema(
            name="db_docs",
            primary_key="chunk_id",
            text_column="chunked_text",
            doc_uri="doc_uri",
        )

        # Load the retriever tool's docs.
        raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
        self.docs = pd.read_parquet(raw_docs_parquet).to_dict("records")

        # Identify the function used as the retriever tool
        self.tool_functions = {
            'search_product_docs': self.search_product_docs
        }

    @mlflow.trace(name="rag_agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> ChatCompletionResponse:
        """
        Primary function that takes a user's request and generates a response.
        """
        # Convert the user's request to a dict
        request = self.get_request_dict(model_input)

        # Add system prompt
        request = {
                **request,
                "messages": [
                    {"role": "system", "content": self.config.get('system_prompt')},
                    *request["messages"],
                ],
            }

        # Ask the LLM to call tools & generate the response
        return self.recursively_call_and_run_tools(
            **request
        )
    
    @mlflow.trace(span_type="RETRIEVER")
    def search_product_docs(self, query: list[str]) -> list[dict]:
        """
        Retriever tool.  Simple keyword-based retriever - would be replaced with a Vector Index
        """
        keywords = query
        if len(keywords) == 0:
            return []
        result = []
        for chunk in self.docs:
            score = sum(
                (keyword.lower() in chunk["chunked_text"].lower())
                for keyword in keywords
            )
            result.append(
                {
                    "page_content": chunk["chunked_text"],
                    "metadata": {
                        "doc_uri": chunk["url"],
                        "score": score,
                        "chunk_id": chunk["chunk_id"],
                    },
                }
            )
        ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
        cutoff_docs = []
        context_budget_left = self.config.get("max_context_chars")
        for doc in ranked_docs:
            content = doc["page_content"]
            doc_len = len(content)
            if context_budget_left < doc_len:
                cutoff_docs.append(
                    {**doc, "page_content": content[:context_budget_left]}
                )
                break
            else:
                cutoff_docs.append(doc)
            context_budget_left -= doc_len
        return cutoff_docs

    ##
    # Helper functions below
    ##
    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Helpers: Call the LLM configured via the ModelConfig using the OpenAI SDK
        """
        traced_chat_completions_create_fn = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )
        request = {**request, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens"),  "tools": RETRIEVER_TOOL_SPEC} #, "parallel_tool_calls":False}
        return traced_chat_completions_create_fn(
            model=self.config.get("endpoint_name"), **request,
                
        )

    @mlflow.trace(span_type="CHAIN")
    def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
        """
        Recursively calls the LLM w/ the tools in the prompt.  Either executes the tools and recalls the LLM or returns the LLM's generation.
        """
        messages = kwargs["messages"]
        del kwargs["messages"]
        i = 0
        while i < max_iter:
            with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
                response = self.chat_completion(request={'messages': messages})
                assistant_message = response.choices[0].message  # openai client
                tool_calls = assistant_message.tool_calls  # openai
                if tool_calls is None:
                    # the tool execution finished, and we have a generation
                    return response.to_dict()
                tool_messages = []
                for tool_call in tool_calls:  # TODO: should run in parallel
                    with mlflow.start_span(
                        name="execute_tool", span_type="TOOL"
                    ) as span:
                        function = tool_call.function  
                        args = json.loads(function.arguments)  
                        span.set_inputs(
                            {
                                "function_name": function.name,
                                "function_args_raw": function.arguments,
                                "function_args_loaded": args,
                            }
                        )
                        result = self.execute_function(
                            self.tool_functions[function.name], args
                        )
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        } 

                        tool_messages.append(tool_message)
                        span.set_outputs({"new_message": tool_message})
                assistant_message_dict = assistant_message.dict().copy()  
                del assistant_message_dict["content"]
                del assistant_message_dict["function_call"] 
                if "audio" in assistant_message_dict:
                    del assistant_message_dict["audio"]  # hack to make llama70b work
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

    def execute_function(self, tool, args):
        """
        Execute a tool and return the result as a JSON string
        """
        result = tool(**args)
        return json.dumps(result)

    @mlflow.trace(span_type="PARSER")
    def get_request_dict(self, 
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
    ) -> List[Dict[str, str]]:
        """
        Since the PyFunc model can get either a dict, list, or pd.DataFrame depending on where it is called from (locally, evaluation, or model serving), unify all requests to a dictionary.
        """
        if type(model_input) == list:
            # get the first row
            model_input = list[0]
        elif type(model_input) == pd.DataFrame:
            # return the first row, this model doesn't support batch input
            return model_input.to_dict(orient="records")[0]
        
        # now, try to unpack the single item or first row of batch input
        if type(model_input) == ChatCompletionRequest:
            return asdict(model_input)
        elif type(model_input) == dict:
            return model_input
        


# tell MLflow logging where to find the agent's code
mlflow.models.set_model(FunctionCallingAgent())
