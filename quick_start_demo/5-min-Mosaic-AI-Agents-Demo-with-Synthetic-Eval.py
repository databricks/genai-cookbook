# Databricks notebook source
# MAGIC %md
# MAGIC # Demo overview
# MAGIC This notebook shows you how to use Mosaic AI to evaluate and improve the quality/cost/latency of a tool-calling Agent, deploying the resulting Agent to a web-based chat UI.
# MAGIC
# MAGIC Using Mosiac AI [Agent Evaluation](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html), [Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/build-genai-apps.html), [MLflow](https://docs.databricks.com/en/generative-ai/agent-framework/log-agent.html) and [Model Serving](https://docs.databricks.com/en/generative-ai/agent-framework/deploy-agent.html), we will:
# MAGIC 1. Generate synthetic evaluation data from a document corpus
# MAGIC 2. Create a tool-calling Agent with a Retriever tool
# MAGIC 3. Evaluate the Agent's quality/cost/latency across several foundational models
# MAGIC 4. Deploy the Agent to a web-based chat app
# MAGIC
# MAGIC Requirements: 
# MAGIC * Use a Serverless or MLR/DBR 14.3+ compute cluster
# MAGIC * Databricks Serverless & Unity Catalog enabled
# MAGIC * CREATE MODEL access to a Unity Catalog schema
# MAGIC * Permission to create Model Serving endpoints
# MAGIC
# MAGIC **How to use this notebook**:
# MAGIC 1. Update the following cell to configure the UC location where the agent model will be register to
# MAGIC
# MAGIC  ✅✏️ CONFIGURE THE UC MODEL NAME HERE
# MAGIC
# MAGIC 2. Click Run all
# MAGIC
# MAGIC ![alt text](https://github.com/databricks/genai-cookbook/blob/demo_gifs/quick_start_demo/demo_gifs/demo_overview.gif?raw=true "Title")
# MAGIC
# MAGIC See this [YouTube channel](https://www.youtube.com/@EricPeter-q6o) if you'd like to watch videos that go deeper into the capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC # 0/ Setup

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow-skinny databricks-sdk[openai]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.1/ Configure user name to avoid conflicts with other users running the demo

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

# Get current user's name & email to ensure each user doesn't over-write other user's outputs
w = WorkspaceClient()
user_email = w.current_user.me().user_name
user_name = user_email.split("@")[0].replace(".", "_")

print(f"User name: {user_name}")

# COMMAND ----------

########################
# #### ✅✏️ CONFIGURE THE UC MODEL NAME HERE
########################
#UC_MODEL_NAME = f"<ENTER_YOUR_CATALOG_NAME>.<ENTER_YOUR_SCHEMA_NAME>.db_docs__{user_name}"
UC_MODEL_NAME = f"yz_agent_demo.playground.db_docs__{user_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1/ Generate synthetic evaluation data to measure quality

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **Challenges Addressed**
# MAGIC 1. How to start quality evaluation with diverse, representative data without SMEs spending months labeling?
# MAGIC
# MAGIC **What is happening?**
# MAGIC - We pass the documents to the Synthetic API along with a `num_evals` and prompt-like `guidelines` to tailor the generated questions for our use case. This API uses a proprietary synthetic generation pipeline developed by Mosaic AI Research.
# MAGIC - The API produces `num_evals` questions, each coupled with the source document & a list of facts, generated based on the source document.  Each fact must be present in the Agent's response for it to be considered correct.
# MAGIC
# MAGIC *Why does the the API generates a list of facts, rather than a fully written answer.  This...*
# MAGIC - Makes SME review more efficient: by focusing on facts rather than a full response, they can review/edit more quickly.
# MAGIC - Improves the accuracy of our proprietary LLM judges.
# MAGIC
# MAGIC Interested in have your SMEs review the data?  Check out a [video demo of the Eval Set UI](https://youtu.be/avY9724q4e4?feature=shared&t=130).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1/ Load the docs corpus from the Cookbook repo

# COMMAND ----------

# MAGIC %md
# MAGIC First, we load the documents (Databricks documentation) used by our Agent, filtering for a subset of the documentation we want to use 

# COMMAND ----------

import pandas as pd

databricks_docs_url = "https://raw.githubusercontent.com/databricks/genai-cookbook/refs/heads/main/quick_start_demo/chunked_databricks_docs_filtered.jsonl"
parsed_docs_df = pd.read_json(databricks_docs_url, lines=True)

display(parsed_docs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2/ Call API to generate synthetic evaluation data

# COMMAND ----------

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for guidelines - feel free to modify as you see fit.
guidelines = f"""
# Task Description
The Agent is a RAG chatbot that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

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
    docs=parsed_docs_df[:500], # Pass your docs. Pandas/Spark Dataframes with columns `content STRING, doc_uri STRING` are suitable.
    num_evals=num_evals, # How many synthetic evaluations to generate
    guidelines=guidelines
)
display(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2/ Write the Agent's code

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1/ Function-calling agent w/ a Retriever Tool

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenges addressed**
# MAGIC - How do I track different versions of my agent's code/config?
# MAGIC - How do I enable observability, monitoring, and debugging of my Agent’s logic?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC First, we create a function-calling Agent with access to a retriever tool using OpenAI SDK + Python code.  To keep the demo simple, the retriever is a function that performs keyword lookup rather than a vector search index.
# MAGIC
# MAGIC When creating your Agent, you can either:
# MAGIC 1. Generate template Agent code from the AI Playground
# MAGIC 2. Use a template from our Cookbook
# MAGIC 3. Start from an example in popular frameworks such as LangGraph, AutoGen, LlamaIndex, and others.
# MAGIC
# MAGIC **NOTE: It is not necessary to understand how this Agent works to understand the rest of this demo notebook.**  
# MAGIC
# MAGIC *A few things to note about the code:*
# MAGIC 1. The code is written to `fc_agent.py` in order to use [MLflow Models from Code](https://www.mlflow.org/blog/models_from_code) for logging, enabling easy tracking of each iteration as we tune the Agent for quality.
# MAGIC 2. The code is parameterized with an [MLflow Model Configuration](https://docs.databricks.com/en/generative-ai/agent-framework/create-agent.html#use-parameters-to-configure-the-agent), enabling easy tuning of these parameters for quality improvement.
# MAGIC 3. The code is wrapped in an MLflow PyFunc model, making the Agent's code deployment-ready so any iteration can be shared with stakeholders for testing.
# MAGIC 4. The code implements [MLflow Tracing](https://docs.databricks.com/en/mlflow/mlflow-tracing.html) for unified observability during development and production. The same trace defined here will be logged for every production request post-deployment. For agent authoring frameworks like LangChain and LlamaIndex, you can tracing with one line of code: `mlflow.langchain.autolog()`/`mlflow.llama_index.autolog()`.

# COMMAND ----------

# MAGIC %%writefile fc_agent.py
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from openai import OpenAI
# MAGIC import pandas as pd
# MAGIC from typing import Any, Union, Dict, List
# MAGIC import mlflow
# MAGIC from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
# MAGIC import dataclasses
# MAGIC import json
# MAGIC
# MAGIC DEFAULT_CONFIG = {
# MAGIC     'endpoint_name': "agents-demo-gpt4o",
# MAGIC     'temperature': 0.01,
# MAGIC     'max_tokens': 1000,
# MAGIC     'system_prompt': """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.
# MAGIC
# MAGIC     You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
# MAGIC     """,
# MAGIC     'max_context_chars': 4096 * 4
# MAGIC }
# MAGIC
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
# MAGIC                     "description": "a set of individual keywords to find relevant docs for.  each item of the array must be a single word.",
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
# MAGIC class FunctionCallingAgent(mlflow.pyfunc.PythonModel):
# MAGIC     """
# MAGIC     Class representing a function-calling Agent that has one tool: a retriever w/ keyword-based search.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self):
# MAGIC         """
# MAGIC         Initialize the OpenAI SDK client connected to Model Serving.
# MAGIC         Load the Agent's configuration from MLflow Model Config.
# MAGIC         """
# MAGIC         # Initialize OpenAI SDK connected to Model Serving
# MAGIC         w = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = w.serving_endpoints.get_open_ai_client()
# MAGIC
# MAGIC         # Load config
# MAGIC         self.config = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
# MAGIC
# MAGIC         # Configure playground & review app & agent evaluation to display / see the chunks from the retriever 
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
# MAGIC         self,
# MAGIC         context: Any = None,
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
# MAGIC         params: Any = None,
# MAGIC     ) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Primary function that takes a user's request and generates a response.
# MAGIC         """
# MAGIC         # Convert the user's request to a dict
# MAGIC         request = self.get_request_dict(model_input)
# MAGIC
# MAGIC         # Add system prompt
# MAGIC         request = {
# MAGIC                 **request,
# MAGIC                 "messages": [
# MAGIC                     {"role": "system", "content": self.config.get('system_prompt')},
# MAGIC                     *request["messages"],
# MAGIC                 ],
# MAGIC             }
# MAGIC
# MAGIC         # Ask the LLM to call tools & generate the response
# MAGIC         return self.recursively_call_and_run_tools(
# MAGIC             **request
# MAGIC         )
# MAGIC     
# MAGIC     @mlflow.trace(span_type="RETRIEVER")
# MAGIC     def search_product_docs(self, query: list[str]) -> list[dict]:
# MAGIC         """
# MAGIC         Retriever tool.  Simple keyword-based retriever - would be replaced with a Vector Index
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
# MAGIC     def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
# MAGIC         """
# MAGIC         Helpers: Call the LLM configured via the ModelConfig using the OpenAI SDK
# MAGIC         """
# MAGIC         traced_chat_completions_create_fn = mlflow.trace(
# MAGIC             self.model_serving_client.chat.completions.create,
# MAGIC             name="chat_completions_api",
# MAGIC             span_type="CHAT_MODEL",
# MAGIC         )
# MAGIC         request = {**request, "temperature": self.config.get("temperature"), "max_tokens": self.config.get("max_tokens"),  "tools": RETRIEVER_TOOL_SPEC} #, "parallel_tool_calls":False}
# MAGIC         return traced_chat_completions_create_fn(
# MAGIC             model=self.config.get("endpoint_name"), **request,
# MAGIC                 
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type="CHAIN")
# MAGIC     def recursively_call_and_run_tools(self, max_iter=10, **kwargs):
# MAGIC         """
# MAGIC         Recursively calls the LLM w/ the tools in the prompt.  Either executes the tools and recalls the LLM or returns the LLM's generation.
# MAGIC         """
# MAGIC         messages = kwargs["messages"]
# MAGIC         del kwargs["messages"]
# MAGIC         i = 0
# MAGIC         while i < max_iter:
# MAGIC             with mlflow.start_span(name=f"iteration_{i}", span_type="CHAIN") as span:
# MAGIC                 response = self.chat_completion(request={'messages': messages})
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
# MAGIC     @mlflow.trace(span_type="PARSER")
# MAGIC     def get_request_dict(self, 
# MAGIC         model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame, List]
# MAGIC     ) -> List[Dict[str, str]]:
# MAGIC         """
# MAGIC         Since the PyFunc model can get either a dict, list, or pd.DataFrame depending on where it is called from (locally, evaluation, or model serving), unify all requests to a dictionary.
# MAGIC         """
# MAGIC         if type(model_input) == list:
# MAGIC             # get the first row
# MAGIC             model_input = list[0]
# MAGIC         elif type(model_input) == pd.DataFrame:
# MAGIC             # return the first row, this model doesn't support batch input
# MAGIC             return model_input.to_dict(orient="records")[0]
# MAGIC         
# MAGIC         # now, try to unpack the single item or first row of batch input
# MAGIC         if type(model_input) == ChatCompletionRequest:
# MAGIC             return asdict(model_input)
# MAGIC         elif type(model_input) == dict:
# MAGIC             return model_input
# MAGIC         
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
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2/ Vibe check the Agent

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test the Agent for a sample query to see the MLflow Trace.

# COMMAND ----------

from fc_agent import FunctionCallingAgent
fc_agent = FunctionCallingAgent()

dev_config = {
    'endpoint_name': "agents-demo-gpt4o",
    'temperature': 0.01,
    'max_tokens': 1000,
    'system_prompt': """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
    'max_context_chars': 4096 * 4
}

mlflow.models.ModelConfig(development_config=dev_config)

response = fc_agent.predict(model_input={"messages": [{"role": "user", "content": "What is lakehouse monitoring?"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC # 3/ Evaluate the Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1/ Initial evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - What are the right metrics to evaluate quality?  How do I trust the outputs of these metrics?
# MAGIC - I need to evaluate many ideas - how do I…
# MAGIC     - …run evaluation quickly so the majority of my time isn’t spent waiting?
# MAGIC     - …quickly compare these different versions of my agent on cost/quality/latency?
# MAGIC - How do I quickly identify the root cause of any quality problems?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC Now, we run Agent Evaluation's propietary LLM judges using the synthetic evaluation set to see the quality/cost/latency of the Agent and identify any root causes of quality issues.  Agent Evaluation is tightly integrated with `mlflow.evaluate()`.  
# MAGIC
# MAGIC Mosaic AI Research has invested signficantly in the quality AND speed of the LLM judges, optimizing the judges to agree with human raters.  Read more [details in our blog](https://www.databricks.com/blog/databricks-announces-significant-improvements-built-llm-judges-agent-evaluation) about how our judges outperform the competition.  
# MAGIC
# MAGIC Once evaluation runs, click `View Evaluation Results` to open the MLflow UI for this Run.  This lets you:
# MAGIC - See summary metrics
# MAGIC - See root cause analysis that identifies the most important issues to fix
# MAGIC - Inspect individual responses to gain intuition about how the Agent is performing
# MAGIC - See the judge outputs to understand why the responses were graded as good/bad
# MAGIC - Compare between multiple runs to see how quality changed between experiments
# MAGIC
# MAGIC You can also inspect the other tabs:
# MAGIC - `Overview` lets you see the Agent's config/parameters
# MAGIC - `Artifacts` lets you see the Agent's code
# MAGIC
# MAGIC This UIs, coupled with the speed of evaluation, help you efficiently test your hypotheses to improve quality, letting you reach the production quality bar in less time. 
# MAGIC
# MAGIC ![alt text](https://github.com/databricks/genai-cookbook/blob/demo_gifs/quick_start_demo/demo_gifs/eval_1.gif?raw=true "Title")

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionResponse, ChatCompletionRequest
from mlflow.models.resources import DatabricksServingEndpoint

# First, we define a helper function so we can compare the agent across multiple parameters & LLMs.
def log_and_evaluate_agent(agent_config: dict, run_name: str):

    # Define the databricks resources so this logged Agent is deployment ready
    resources = [DatabricksServingEndpoint(endpoint_name=agent_config["endpoint_name"])]

    # Start a run to contain the Agent.  `run_name` is a human-readable label for this run.
    with mlflow.start_run(run_name=run_name):
        # Log the Agent's code/config to MLflow
        model_info = mlflow.pyfunc.log_model(
            python_model="fc_agent.py",
            artifact_path="agent",
            model_config=agent_config,
            signature=ModelSignature(
                inputs=ChatCompletionRequest(),
                outputs=ChatCompletionResponse(),
            ),
            resources=resources,
            input_example={"messages": [{"role": "user", "content": "What is lakehouse monitoring?"}]},
            pip_requirements=["databricks-sdk[openai]", "mlflow", "databricks-agents"],
        )

        # Run evaluation
        eval_results = mlflow.evaluate(
            data=evals,  # Your evaluation set
            model=model_info.model_uri,  # Logged Agent from above
            model_type="databricks-agent",  # activate Mosaic AI Agent Evaluation
        )

        return (model_info, eval_results)


# Now we call the helper function to run evaluation.
# The configuration keys must match those defined in `fc_agent.py`
model_info_gpt4o, eval_results = log_and_evaluate_agent(
    agent_config={
        "endpoint_name": "agents-demo-gpt4o",
        "temperature": 0.01,
        "max_tokens": 1000,
        "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
        "max_context_chars": 4096 * 4,
    },
    run_name="gpt-4o",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2/ Compare multiple LLMs on quality/cost/latency

# COMMAND ----------

# MAGIC %md
# MAGIC **Challenges addressed**
# MAGIC - How to determine the foundational model that offers the right balance of quality/cost/latency?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC For the purposes of the demo, let's assume we have fixed any root causes identified above and want to optimize our Agent for quality/cost/latency.  
# MAGIC
# MAGIC Here we will run evaluation for several LLMs.  Once evaluation runs, click `View Evaluation Results` to open the MLflow UI for one of the runs.  In the MLFLow Evaluations UI, use the "Compare to Run" dropdown to select another run name.  This comparison view helps you quickly identify where the Agent got better/worse/stayed the same.
# MAGIC
# MAGIC Then, go to the MLflow Experiement page and click the chart icon in the upper left corner by `Runs`.  Here, you can compare the models quantiatively across quality/cost/latency metrics.  Cost is proxied through the number of tokens used.
# MAGIC
# MAGIC This helps you make informed tradeoffs in partnership with your business stakeholders about cost/latency/quality.  Further, you can use this view to provide quantitative updates to your stakeholders so they can follow your progress improving quality!
# MAGIC
# MAGIC ![alt text](https://github.com/databricks/genai-cookbook/blob/demo_gifs/quick_start_demo/demo_gifs/eval_2.gif?raw=true "Title")

# COMMAND ----------

baseline_config = {
    "endpoint_name": "agents-demo-gpt4o",
    "temperature": 0.01,
    "max_tokens": 1000,
    "system_prompt": """You are a helpful assistant that answers questions about Databricks. Questions unrelated to Databricks are irrelevant.

    You answer questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.
    """,
    "max_context_chars": 4096 * 4,
}

llama70b_config = baseline_config.copy()
llama70b_config['endpoint_name'] = 'databricks-meta-llama-3-1-70b-instruct'
model_info_llama70b, _ = log_and_evaluate_agent(
    agent_config=llama70b_config,
    run_name="llama70b",
)

gpt_4o_mini_config = baseline_config.copy()
gpt_4o_mini_config['endpoint_name'] = 'agents-demo-gpt4o-mini'

model_info_gpt4o_mini, _ = log_and_evaluate_agent(
    agent_config=gpt_4o_mini_config,
    run_name="gpt_4o_mini",
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4/ [Optional] Deploy the RAG Agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1/ Deploy to pre-production for stakeholder testing
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - How do I quickly create a Chat UI for stakeholders to test the agent?
# MAGIC - How do I track each piece of feedback and have it linked to what is happening in the bot so I can debug issues – without resorting to spreadsheets?
# MAGIC
# MAGIC **What is happening?**
# MAGIC
# MAGIC First, we register one of the Agent models that we logged above to the Unity Catalog.  Then, we use Agent Framework to deploy the Agent to Model serving using one line of code: `agents.deploy()`.
# MAGIC
# MAGIC The resulting Model Serving endpoint:
# MAGIC - Is connected to the Review App, which is a lightweight chat UI that can be shared with any user in your company, even if they don't have Databricks workspace access
# MAGIC - Is integrated with AI Gateway so every request/response and its accompanying MLflow trace and user feedback is stored in an Inference Table
# MAGIC
# MAGIC Optionally, you could turn on Agent Evaluation’s monitoring capabilities, which are unified with the offline experience we used above, and get a ready-to-go dashboard that runs judges on a sample of the traffic.
# MAGIC
# MAGIC ![alt text](https://github.com/databricks/genai-cookbook/blob/demo_gifs/quick_start_demo/demo_gifs/review_app.gif?raw=true "Title")

# COMMAND ----------

from databricks import agents

# Connect to the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Configure UC model location
# UC_MODEL_NAME = f"agents_demo.synthetic_data.db_docs__{user_name}"

# Register the gpt-4o version to Unity Catalog 
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info_gpt4o.model_uri, name=UC_MODEL_NAME
)
# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(
    model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version
)

# Wait for the Review App & REST API to be ready
print("Wait for endpoint to deploy.  This can take 15 - 20 minutes.")
print(f"Endpoint name: {deployment_info.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5/ Deploy to production & monitor

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Challenges addressed**
# MAGIC - How do I host my Agent as a production ready, scalable service?
# MAGIC - How do I execute tool code securely and ensure it respects my governance policies?
# MAGIC - How do I enable telemetry / observability in development and production?
# MAGIC - How do I monitor my Agent’s quality at-scale in production?  How do I quickly investigate and fix any quality issues?
# MAGIC
# MAGIC With Agent Framework, production deployment is the same for pre-production & production - you already have a highly scalable REST API that can be intergated in your application.  This API provides an endpoint to get Agent responses and to pass back user feedback so you can use that feedback to improve quality.
# MAGIC
# MAGIC To learn more about how monitoring works (TLDR; we've adapted a version of the above UIs and LLM judges for monitoring), read our [documentation](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluating-production-traffic.html) or watch this [2 minute video](https://www.youtube.com/watch?v=ldAzmKkvQTU).

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Steps
# MAGIC
# MAGIC Use your own data to build agents with Mosaic AI. Navigate to [this notebook](placeholder)

# COMMAND ----------


