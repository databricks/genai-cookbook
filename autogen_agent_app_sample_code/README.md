# How to use local IDE

- databricks auth profile DEFAULT is set up
```
databricks auth profile login
```
- add a cluster_id in ~/.databrickscfg (if you want to use Spark code)
- add `openai_sdk_agent_app_sample_code/.env` to point to mlflow exp + dbx tracking uri (if you want to run any agent code from the terminal and have it logged to mlflow).  Make sure this mlflow experiment maps to the one in 02_agent_setup.ipynb.
```
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_NAME=/Users/your.name@company.com/my_agent_mlflow_experiment
```
- install poetry env & activate in your IDE
```
poetry install
```

if you want to use the data pipeline code in spark, you need to build the cookbook wheel and install it in the cluster
- build cookbook wheel
```
poetry build
```
- install cookbook wheel in cluster
    - Copy the wheel file to a UC Volume or Workspace folder
    - Go to the cluster's Libraries page and install the wheel file as a new library


# NOTES/what doesn't work:
- Works locally & once deployed:
    - Tool calling agent with vector search, UC tool
- Works locally, deployment not tested.
    - Genie Agent
    - Multi-Agent supervisor w/ "local" mode
    - Multi-Agent supervisor w/ "endpoint" mode

# TODO:
- Refactor the cookbook folder to 
    - make it easy to add as `code_path` without putting all agent code + data pipeline code into the agent mlflow model
    - make the data pipeline competely seperate
    - make the tools inherit from a version of serializableConfig that is "serializableTool" - same exact thing just not overloaded.

- Multi-agent
    - test with deployed endpoints
    - make deployed endpoint optional if model = local, otherwise, make class/config optional.

- Create a version of each of these Agents with LangGraph, LlamaIndex, and AutoGen.

# Docs

This cookbook contains example Agents built using Python code + the OpenAI SDK to call Databricks Model Serving/External Models.  Each Agent is configurable via a Pydantic-based configuration classes and is wrapped in an MLflow PyFunc class for logging and deployment.

Included are 3 types of Agents:
- Tool Calling Agent
- Genie Agent
- Multi-Agent Supervisor Agent

## Genie Agent

The Genie Agent is a simple wrapper around AI/BI Genie Spaces API.  It does not use the OpenAI SDK.  It is configured using the `GenieAgentConfig` class:
- Required
    - `genie_space_id: str` - The ID of the Genie Space
- Optional Variables with Default Values
    - `input_example: Any` - Defaults to:
    ```python
    {
        "messages": [
            {
                "role": "user",
                "content": "What types of data can I query?",
            },
        ]
    }
    ```
    - `encountered_error_user_message: str` - Defaults to:
    > "I encountered an error trying to answer your question, please try again."

## Tool-calling Agent

The tool-calling agent uses the configured LLM to decide which tool(s) to call based on the user's query.  

The agent is configured using the `FunctionCallingAgentConfig` class:

- Required:
    - `llm_config: LLMConfig` - Configuration for the LLM endpoint
    - `tools: List[BaseTool]` - List of tools available to the agent.  

- Optional Variables with Default Values:
    - `input_example: Any` - Defaults to:
    ```python
    {
        "messages": [
            {
                "role": "user",
                "content": "What can you help me with?",
            },
        ]
    }
    ```

The `LLMConfig` requires:
- `llm_endpoint_name: str` - Name of the model serving endpoint
- `llm_system_prompt_template: str` - System prompt for the LLM
- `llm_parameters: Dict` - Parameters for the LLM (temperature, max_tokens, etc.)

The `BaseTool` class is used to define a tool that the agent can call.  The cookbook includes several pre-built tools.  If you need to create your own tool, we suggest creating a UC Function and calling that function using the `UCTool`. 
- UC Tool
    - Wraps the uc toolkit.  Adds add't code to parse errors from spark exceptions to just show the Python errors.
- Vector Search Retriever Tool
    - A 




## How Pydantic configuration classes work
All configuration classes inherit from `SerializableConfig`, defined in `config/__init__.py`.  This class enables a Pydantic BaseModel to be serialized to a YAML file and loaded back from that YAML file.



The Genie Agent is a simple wrapper around AI/BI Genie Spaces API.  It does not use the OpenAI SDK.



s code is wrapped in an MLflow PyFunc class and 


UC Function Tool

Local Function Tool

## Vector Search Retriever Tool

Issues
- Vector Search index does not store the source table's column name / description metadata, so the tool currently uses the source table's metadata to populate the filterable columns.  However, this causes deployment to fail since the deployed model does not have access to the source table, so it is toggled off by `USE_SOURCE_TABLE_FOR_METADATA`.


Features:
* User can specify a list of filterable columns; these are presented to the tool-calling LLM as parameters of the tool.


* Validates all provided columns exist




what do you need to do?

- make your data pipeline
- create your genie spaces
- create your tools
- create your agents
- create your multi-agent supervisor



create a unsutrcutred data agent
- create data pipeline
- create synthetic data
- create agent with retriever tool
- evaluate and iterate
- maybe add some tools
