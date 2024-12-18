# Databricks notebook source
# MAGIC %md
# MAGIC # Demo overview
# MAGIC This notebook shows you how to use Mosaic AI to evaluate and improve the quality, cost, and latency of a tool-calling agent, deploying the resulting agent to a web-based chat UI.
# MAGIC
# MAGIC Using Mosiac AI Agent Evaluation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/)), Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/agent-framework/build-genai-apps.html) |[Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/build-genai-apps)), MLflow ([AWS](https://docs.databricks.com/en/generative-ai/agent-framework/log-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/log-agent)) and Model Serving ([AWS](https://docs.databricks.com/en/generative-ai/agent-framework/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/deploy-agent)), this notebook:
# MAGIC 1. Generates synthetic evaluation data from a document corpus.
# MAGIC 2. Creates a tool-calling agent with a retriever tool.
# MAGIC 3. Evaluates the agent's quality, cost, and latency across several foundational models.
# MAGIC 4. Deploys the agent to a web-based chat app.
# MAGIC
# MAGIC Requirements: 
# MAGIC * Use serverless compute or a cluster running Databricks Runtime 14.3 or above.
# MAGIC * Databricks Serverless and Unity Catalog enabled.
# MAGIC * CREATE MODEL access to a Unity Catalog schema.
# MAGIC * Permission to create Model Serving endpoints.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/generative-ai/synth-evals/demo-overview-optimized.gif"/>
# MAGIC
# MAGIC For videos that go deeper into the capabilities, see this [YouTube channel](https://www.youtube.com/@EricPeter-q6o).

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

#%pip install -U -qqqq databricks-agents mlflow databricks-sdk[openai] backoff
%pip install -U -qqqq mlflow databricks-sdk[openai] "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-studio/staging/databricks_agents-0.11.1.dev1-py3-none-any.whl" backoff

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %reload_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1. Generate synthetic evaluation data to measure quality

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **Challenges Addressed**
# MAGIC 1. How to start quality evaluation with diverse, representative data without SMEs spending months labeling?
# MAGIC
# MAGIC **What is happening?**
# MAGIC - We pass the documents to the Synthetic API along with a `num_evals` and prompt-like `agent_description` and `question_guidelines` to tailor the generated questions for our use case. This API uses a proprietary synthetic generation pipeline developed by Mosaic AI Research.
# MAGIC - The API produces `num_evals` questions, each coupled with the source document and a list of facts, generated based on the source document. Each fact must be present in the agent's response for it to be considered correct.
# MAGIC
# MAGIC *Why does the the API generates a list of facts, rather than a fully written answer. This...*
# MAGIC - Makes SME review more efficient: by focusing on facts rather than a full response, they can review and edit more quickly.
# MAGIC - Improves the accuracy of our proprietary LLM judges.
# MAGIC
# MAGIC Interested in have your SMEs review the data? Check out a [video demo of the Eval Set UI](https://youtu.be/avY9724q4e4?feature=shared&t=130).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the docs corpus

# COMMAND ----------

# MAGIC %md
# MAGIC First, load the documents (Databricks documentation) used by the agent, filtering for a subset of the documentation.
# MAGIC
# MAGIC For your agent, replace this step to instead load your parsed documents.

# COMMAND ----------

import pandas as pd

databricks_docs_url = "https://raw.githubusercontent.com/databricks/genai-cookbook/refs/heads/main/quick_start_demo/chunked_databricks_docs_filtered.jsonl"
parsed_docs_df = pd.read_json(databricks_docs_url, lines=True)

display(parsed_docs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call API to generate synthetic evaluation data

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
agent_description = f"""
The agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
"""
question_guidelines = f"""
# User personas
- A developer who is new to the Databricks platform
- An experienced, highly technical Data Scientist or Data Engineer

# Example questions
- what API lets me parallelize operations over rows of a delta table?
- Which cluster settings will give me the best performance when using Spark?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 25
evals = generate_evals_df(
    docs=parsed_docs_df[
        :500
    ],  # Pass your docs. They should be in a Pandas or Spark DataFrame with columns `content STRING` and `doc_uri STRING`.
    num_evals=num_evals,  # How many synthetic evaluations to generate
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)
display(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2. Write the agent's code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function-calling agent that uses a retriever tool

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenges addressed**
# MAGIC - How do I track different versions of my agent's code or configuration?
# MAGIC - How do I enable observability, monitoring, and debugging of my agent’s logic?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC First, create a function-calling agent with access to a retriever tool using OpenAI SDK and Python code. To keep the demo simple, the retriever is a function that performs keyword lookup rather than a vector search index.
# MAGIC
# MAGIC When creating your agent, you can either:
# MAGIC 1. Generate template agent code from the AI Playground
# MAGIC 2. Use a template from our Cookbook
# MAGIC 3. Start from an example in popular frameworks such as LangGraph, AutoGen, LlamaIndex, and others.
# MAGIC
# MAGIC **NOTE: It is not necessary to understand how this agent works to understand the rest of this demo notebook.**  
# MAGIC
# MAGIC *A few things to note about the code:*
# MAGIC 1. The code is written to `fc_agent.py` in order to use [MLflow Models from Code](https://www.mlflow.org/blog/models_from_code) for logging, enabling easy tracking of each iteration as you tune the agent for quality.
# MAGIC 2. The code is parameterized with an MLflow Model Configuration ([AWS](https://docs.databricks.com/en/generative-ai/agent-framework/create-agent.html#use-parameters-to-configure-the-agent) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/create-agent#agent-parameters)), enabling easy tuning of these parameters for quality improvement.
# MAGIC 3. The code is wrapped in an MLflow [ChatModel](https://mlflow.org/docs/latest/llms/chat-model-intro/index.html), making the agent's code deployment-ready so any iteration can be shared with stakeholders for testing.
# MAGIC 4. The code implements MLflow Tracing ([AWS](https://docs.databricks.com/en/mlflow/mlflow-tracing.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/mlflow-tracing)) for unified observability during development and production. The same trace defined here will be logged for every production request post-deployment. For agent authoring frameworks, you can tracing with one line of code: `mlflow.framework_agent.autolog()`.

# COMMAND ----------

# MAGIC %%writefile fc_agent.py
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC import openai
# MAGIC import pandas as pd
# MAGIC from typing import Any, Union, Dict, List, Optional
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ChatModel
# MAGIC from mlflow.types.llm import ChatResponse, ChatMessage, ChatParams, ChatChoice
# MAGIC from dataclasses import asdict
# MAGIC import dataclasses
# MAGIC import json
# MAGIC import backoff  # for exponential backoff on LLM rate limits
# MAGIC
# MAGIC
# MAGIC # Default configuration for the agent.
# MAGIC DEFAULT_CONFIG = {
# MAGIC     'endpoint_name': "databricks-meta-llama-3-1-70b-instruct",
# MAGIC     'temperature': 0.01,
# MAGIC     'max_tokens': 1000,
# MAGIC     'system_prompt': """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
# MAGIC
# MAGIC     You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
# MAGIC     """,
# MAGIC     'max_context_chars': 4096 * 4
# MAGIC }
# MAGIC
# MAGIC # OpenAI-formatted function for the retriever tool
# MAGIC RETRIEVER_TOOL_SPEC = [{
# MAGIC     "type": "function",
# MAGIC     "function": {
# MAGIC         "name": "search_product_docs",
# MAGIC         "description": "Use this tool to search for Databricks product documentation.",
# MAGIC         "parameters": {
# MAGIC             "type": "object",
# MAGIC             "required": ["query"],
# MAGIC             "additionalProperties": False,
# MAGIC             "properties": {
# MAGIC                 "query": {
# MAGIC                     "description": "a set of individual keywords to find relevant docs for. each item of the array must be a single word.",
# MAGIC                     "type": "array",
# MAGIC                     "items": {
# MAGIC                         "type": "string"
# MAGIC                     }
# MAGIC                 }
# MAGIC             },
# MAGIC         },
# MAGIC     },
# MAGIC }]
# MAGIC
# MAGIC class FunctionCallingAgent(mlflow.pyfunc.ChatModel):
# MAGIC     """
# MAGIC     Class representing a function-calling agent that has one tool: a retriever using keyword-based search.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         """
# MAGIC         Initialize the OpenAI SDK client connected to Model Serving.
# MAGIC         Load the agent's configuration from MLflow Model Config.
# MAGIC         """
# MAGIC         # Initialize OpenAI SDK connected to Model Serving
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load config
# MAGIC         # When this agent is deployed to Model Serving, the configuration loaded here is replaced with the config passed to mlflow.pyfunc.log_model(model_config=...)
# MAGIC         self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
# MAGIC
# MAGIC         # Configure playground, review app, and agent evaluation to display the chunks from the retriever 
# MAGIC         mlflow.models.set_retriever_schema(
# MAGIC             name="db_docs",
# MAGIC             primary_key="chunk_id",
# MAGIC             text_column="chunked_text",
# MAGIC             doc_uri="doc_uri",
# MAGIC         )
# MAGIC
# MAGIC         # Load the retriever tool's docs.
# MAGIC         raw_docs_parquet = "https://github.com/databricks/genai-cookbook/raw/refs/heads/main/quick_start_demo/chunked_databricks_docs.snappy.parquet"
# MAGIC         self.docs = pd.read_parquet(raw_docs_parquet).to_dict("records")
# MAGIC
# MAGIC         # Identify the function used as the retriever tool
# MAGIC         self.tool_functions = {
# MAGIC             'search_product_docs': self.search_product_docs
# MAGIC         }
# MAGIC
# MAGIC     @mlflow.trace(name="rag_agent", span_type="AGENT")
# MAGIC     def predict(
# MAGIC         self, context=None, messages: List[ChatMessage]=None, params: Optional[ChatParams] = None
# MAGIC     ) -> ChatResponse:
# MAGIC         """
# MAGIC         Primary function that takes a user's request and generates a response.
# MAGIC         """
# MAGIC         if messages is None:
# MAGIC             raise ValueError("predict(...) called without `messages` parameter.")
# MAGIC         
# MAGIC         # Convert all input messages to dict from ChatMessage
# MAGIC         messages = convert_chat_messages_to_dict(messages)
# MAGIC
# MAGIC         # Add system prompt
# MAGIC         request = {
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": self.config.get('system_prompt')},
# MAGIC                     *messages,
# MAGIC                 ],
# MAGIC             }
# MAGIC             
# MAGIC         # Ask the LLM to call tools and generate the response
# MAGIC         output= self.recursively_call_and_run_tools(
# MAGIC             **request
# MAGIC         )
# MAGIC         
# MAGIC         # Convert response to ChatResponse dataclass
# MAGIC         return ChatResponse.from_dict(output)
# MAGIC     
# MAGIC     @mlflow.trace(span_type="RETRIEVER")
# MAGIC     def search_product_docs(self, query: list[str]) -> list[dict]:
# MAGIC         """
# MAGIC         Retriever tool. Simple keyword-based retriever - would be replaced with a Vector Index
# MAGIC         """
# MAGIC         keywords = query
# MAGIC         if len(keywords) == 0:
# MAGIC             return []
# MAGIC         result = []
# MAGIC         for chunk in self.docs:
# MAGIC             score = sum(
# MAGIC                 (keyword.lower() in chunk["chunked_text"].lower())
# MAGIC                 for keyword in keywords
# MAGIC             )
# MAGIC             result.append(
# MAGIC                 {
# MAGIC                     "page_content": chunk["chunked_text"],
# MAGIC                     "metadata": {
# MAGIC                         "doc_uri": chunk["url"],
# MAGIC                         "score": score,
# MAGIC                         "chunk_id": chunk["chunk_id"],
# MAGIC                     },
# MAGIC                 }
# MAGIC             )
# MAGIC         ranked_docs = sorted(result, key=lambda x: x["metadata"]["score"], reverse=True)
# MAGIC         cutoff_docs = []
# MAGIC         context_budget_left = self.config.get("max_context_chars")
# MAGIC         for doc in ranked_docs:
# MAGIC             content = doc["page_content"]
# MAGIC             doc_len = len(content)
# MAGIC             if context_budget_left < doc_len:
# MAGIC                 cutoff_docs.append(
# MAGIC                     {**doc, "page_content": content[:context_budget_left]}
# MAGIC                 )
# MAGIC                 break
# MAGIC             else:
# MAGIC                 cutoff_docs.append(doc)
# MAGIC             context_budget_left -= doc_len
# MAGIC         return cutoff_docs
# MAGIC
# MAGIC     ##
# MAGIC     # Helper functions below
# MAGIC     ##
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     def completions_with_backoff(self, **kwargs):
# MAGIC         """
# MAGIC         Helper: exponetially backoff if the LLM's rate limit is exceeded.
# MAGIC         """
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         return traced_chat_completions_create_fn(**kwargs)
# MAGIC
# MAGIC     def chat_completion(self, messages: List[ChatMessage]) -> ChatResponse:
# MAGIC         """
# MAGIC         Helper: Call the LLM configured via the ModelConfig using the OpenAI SDK
# MAGIC         """
# MAGIC         request = {"messages": messages, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens"),  "tools": RETRIEVER_TOOL_SPEC}
# MAGIC         return self.completions_with_backoff(
# MAGIC             model=self.config.get("endpoint_name"), **request,
# MAGIC                 
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type="CHAIN")
# MAGIC     def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
# MAGIC         """
# MAGIC         Helper: Recursively calls the LLM using the tools in the prompt. Either executes the tools and recalls the LLM or returns the LLM's generation.
# MAGIC         """
# MAGIC         messages = kwargs["messages"]
# MAGIC         del kwargs["messages"]
# MAGIC         i = 0
# MAGIC         while i < max_iter:
# MAGIC             with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
# MAGIC                 response = self.chat_completion(messages=messages)
# MAGIC                 assistant_message = response.choices[0].message  # openai client
# MAGIC                 tool_calls = assistant_message.tool_calls  # openai
# MAGIC                 if tool_calls is None:
# MAGIC                     # the tool execution finished, and we have a generation
# MAGIC                     return response.to_dict()
# MAGIC                 tool_messages = []
# MAGIC                 for tool_call in tool_calls:  # TODO: should run in parallel
# MAGIC                     with mlflow.start_span(
# MAGIC                         name="execute_tool", span_type="TOOL"
# MAGIC                     ) as span:
# MAGIC                         function = tool_call.function  
# MAGIC                         args = json.loads(function.arguments)  
# MAGIC                         span.set_inputs(
# MAGIC                             {
# MAGIC                                 "function_name": function.name,
# MAGIC                                 "function_args_raw": function.arguments,
# MAGIC                                 "function_args_loaded": args,
# MAGIC                             }
# MAGIC                         )
# MAGIC                         result = self.execute_function(
# MAGIC                             self.tool_functions[function.name], args
# MAGIC                         )
# MAGIC                         tool_message = {
# MAGIC                             "role": "tool",
# MAGIC                             "tool_call_id": tool_call.id,
# MAGIC                             "content": result,
# MAGIC                         } 
# MAGIC
# MAGIC                         tool_messages.append(tool_message)
# MAGIC                         span.set_outputs({"new_message": tool_message})
# MAGIC                 assistant_message_dict = assistant_message.dict().copy()  
# MAGIC                 del assistant_message_dict["content"]
# MAGIC                 del assistant_message_dict["function_call"] 
# MAGIC                 if "audio" in assistant_message_dict:
# MAGIC                     del assistant_message_dict["audio"]  # hack to make llama70b work
# MAGIC                 messages = (
# MAGIC                     messages
# MAGIC                     + [
# MAGIC                         assistant_message_dict,
# MAGIC                     ]
# MAGIC                     + tool_messages
# MAGIC                 )
# MAGIC                 i += 1
# MAGIC         # TODO: Handle more gracefully
# MAGIC         raise "ERROR: max iter reached"
# MAGIC
# MAGIC     def execute_function(self, tool, args):
# MAGIC         """
# MAGIC         Execute a tool and return the result as a JSON string
# MAGIC         """
# MAGIC         result = tool(**args)
# MAGIC         return json.dumps(result)
# MAGIC         
# MAGIC def convert_chat_messages_to_dict(messages: List[ChatMessage]):
# MAGIC     new_messages = []
# MAGIC     for message in messages:
# MAGIC         if type(message) == ChatMessage:
# MAGIC             # Remove any keys with None values
# MAGIC             new_messages.append({k: v for k, v in asdict(message).items() if v is not None})
# MAGIC         else:
# MAGIC             new_messages.append(message)
# MAGIC     return new_messages
# MAGIC     
# MAGIC
# MAGIC # tell MLflow logging where to find the agent's code
# MAGIC mlflow.models.set_model(FunctionCallingAgent())

# COMMAND ----------

# MAGIC %md
# MAGIC Empty `__init__.py` to allow the `FunctionCallingAgent()` to be imported.

# COMMAND ----------

# MAGIC %%writefile __init__.py
# MAGIC
# MAGIC # Empty file

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vibe check the agent

# COMMAND ----------

# MAGIC %md
# MAGIC Test the agent for a sample query to see the MLflow Trace.

# COMMAND ----------

from fc_agent import FunctionCallingAgent
fc_agent = FunctionCallingAgent()

response = fc_agent.predict(messages=[{"role": "user", "content": "What is lakehouse monitoring?"}])

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3. Evaluate the agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - What are the right metrics to evaluate quality? How do I trust the outputs of these metrics?
# MAGIC - I need to evaluate many ideas - how do I…
# MAGIC     - …run evaluation quickly so the majority of my time isn’t spent waiting?
# MAGIC     - …quickly compare these different versions of my agent on quality, cost, and latency?
# MAGIC - How do I quickly identify the root cause of any quality problems?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC Now, run Agent Evaluation's proprietary LLM judges using the synthetic evaluation set to see the quality, cost, and latency of the agent and identify any root causes of quality issues. Agent Evaluation is tightly integrated with `mlflow.evaluate()`. 
# MAGIC
# MAGIC Mosaic AI Research has invested signficantly in the quality AND speed of the LLM judges, optimizing the judges to agree with human raters. Read more [details in our blog](https://www.databricks.com/blog/databricks-announces-significant-improvements-built-llm-judges-agent-evaluation) about how our judges outperform the competition. 
# MAGIC
# MAGIC After evaluation runs, click `View Evaluation Results` to open the MLflow UI for this Run. This lets you:
# MAGIC - See summary metrics
# MAGIC - See root cause analysis that identifies the most important issues to fix
# MAGIC - Inspect individual responses to gain intuition about how the agent is performing
# MAGIC - See the judge outputs to understand why the responses were graded as pass or fail
# MAGIC - Compare between multiple runs to see how quality changed between experiments
# MAGIC
# MAGIC You can also inspect the other tabs:
# MAGIC - `Overview` lets you see the agent's configuration and parameters
# MAGIC - `Artifacts` lets you see the agent's code
# MAGIC
# MAGIC This UIs, coupled with the speed of evaluation, help you efficiently test your hypotheses to improve quality, letting you reach the production quality bar in less time. 
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/generative-ai/synth-evals/eval-1-optimized.gif"/>

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint
import mlflow

# First, define a helper function so you can compare the agent across multiple parameters and LLMs.
def log_and_evaluate_agent(agent_config: dict, run_name: str):

    # Define the databricks resources so this logged agent is deployment ready
    resources = [DatabricksServingEndpoint(endpoint_name=agent_config["endpoint_name"])]

    # Start a run to contain the agent. `run_name` is a human-readable label for this run.
    with mlflow.start_run(run_name=run_name):
        # Log the agent's code and configuration to MLflow
        model_info = mlflow.pyfunc.log_model(
            python_model="fc_agent.py",
            artifact_path="agent",
            model_config=agent_config,
            resources=resources,
            input_example={
                "messages": [
                    {"role": "user", "content": "What is lakehouse monitoring?"}
                ]
            },
            pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents", "backoff"],
        )

        # Run evaluation
        eval_results = mlflow.evaluate(
            data=evals,  # Your evaluation set
            model=model_info.model_uri,  # Logged agent from above
            model_type="databricks-agent",  # activate Mosaic AI Agent Evaluation
        )

        return (model_info, eval_results)


# Now, call the helper function to run evaluation.
# The configuration keys must match those defined in `fc_agent.py`
model_info_llama_70b, eval_results = log_and_evaluate_agent(
    agent_config={
        "endpoint_name": "databricks-meta-llama-3-1-70b-instruct",
        "temperature": 0.01,
        "max_tokens": 1000,
        "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
        "max_context_chars": 4096 * 4,
    },
    run_name="llama-3-1-70b-instruct",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare multiple LLMs on quality, cost, and latency

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenges addressed**
# MAGIC - How to determine the foundational model that offers the right balance of quality, cost, and latency?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC Normally, you would use the evaluation results to inform your hypotheses to improve quality, iteratively implementing, evaluating, and comparing each idea to the baseline. This demo assumes that you have fixed any root causes identified above and now want to optimize the agent for quality, cost, and latency. 
# MAGIC
# MAGIC Here, you run evaluation for several LLMs. After the evaluation runs, click `View Evaluation Results` to open the MLflow UI for one of the runs. In the MLFLow Evaluations UI, use the **Compare to Run** dropdown to select another run name. This comparison view helps you quickly identify where the agent got better, worse, or stayed the same.
# MAGIC
# MAGIC Then, go to the MLflow Experiement page and click the chart icon in the upper left corner by `Runs`. Here, you can compare the models quantiatively across quality, cost, and latency metrics. The number of tokens used serves as a proxy for cost.
# MAGIC
# MAGIC This helps you make informed tradeoffs in partnership with your business stakeholders about quality, cost, and latency. Further, you can use this view to provide quantitative updates to your stakeholders so they can follow your progress improving quality.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/generative-ai/synth-evals/eval-2-optimized.gif"/>

# COMMAND ----------

baseline_config = {
    "endpoint_name": "databricks-meta-llama-3-1-70b-instruct",
    "temperature": 0.01,
    "max_tokens": 1000,
    "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
    "max_context_chars": 4096 * 4,
}

llama405b_config = baseline_config.copy()
llama405b_config["endpoint_name"] = "databricks-meta-llama-3-1-405b-instruct"
llama405b_config, _ = log_and_evaluate_agent(
    agent_config=llama405b_config,
    run_name="llama-3-1-405b-instruct",
)

# If you have an External Model, such as OpenAI, uncomment this code, and replace `<my-external-model-endpoint-name>` to include this model in the evaluation
# my_model_config = baseline_config.copy()
# my_model_config['endpoint_name'] = '<my-external-model-endpoint-name>'

# model_info_my_model_config, _ = log_and_evaluate_agent(
#     agent_config=my_model_config,
#     run_name=my_model_config['endpoint_name'],
# )

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4. [Optional] Deploy the agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to pre-production for stakeholder testing
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - How do I quickly create a Chat UI for stakeholders to test the agent?
# MAGIC - How do I track each piece of feedback and have it linked to what is happening in the bot so I can debug issues – without resorting to spreadsheets?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC First, register one of the agent models that you logged above to Unity Catalog. Then, use Agent Framework to deploy the agent to Model serving using one line of code: `agents.deploy()`.
# MAGIC
# MAGIC The resulting Model Serving endpoint:
# MAGIC - Is connected to the review app, which is a lightweight chat UI that can be shared with any user in your company, even if they don't have Databricks workspace access
# MAGIC - Is integrated with AI Gateway so every request and response and its accompanying MLflow trace and user feedback is stored in an Inference Table
# MAGIC
# MAGIC Optionally, you can turn on Agent Evaluation’s monitoring capabilities, which are unified with the offline experience used above, and get a ready-to-go dashboard that runs judges on a sample of the traffic.
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/generative-ai/synth-evals/review-app-optimized.gif"/>

# COMMAND ----------

from databricks import agents
import mlflow

# Connect to the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Configure UC model location
UC_MODEL_NAME = f"catalog.schema.db_docs_agent"  # REPLACE WITH UC CATALOG/SCHEMA THAT YOU HAVE `CREATE MODEL` permissions in
assert (
    UC_MODEL_NAME != "catalog.schema.db_docs_agent"
), "Please replace 'catalog.schema.db_docs_agent' with your actual UC catalog and schema."

# Register the Llama 70b version to Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info_llama_70b.model_uri, name=UC_MODEL_NAME
)
# Deploy to enable the review app and create an API endpoint
deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5. Deploy to production and monitor

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - How do I host my agent as a production ready, scalable service?
# MAGIC - How do I execute tool code securely and ensure it respects my governance policies?
# MAGIC - How do I enable telemetry or observability in development and production?
# MAGIC - How do I monitor my agent’s quality at-scale in production? How do I quickly investigate and fix any quality issues?
# MAGIC
# MAGIC With Agent Framework, production deployment is the same for pre-production and production - you already have a highly scalable REST API that can be intergated in your application. This API provides an endpoint to get agent responses and to pass back user feedback so you can use that feedback to improve quality.
# MAGIC
# MAGIC To learn more about how monitoring works (in summary, Databricks has adapted a version of the above UIs and LLM judges for monitoring), read the documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluating-production-traffic.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/evaluating-production-traffic)) or watch this [2 minute video](https://www.youtube.com/watch?v=ldAzmKkvQTU).
