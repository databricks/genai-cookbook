# Agent Application Sample Code

This directory contains sample code for building agent applications using client-side tools. The code demonstrates how to build, evaluate, and improve the quality of your agent applications.

## Directory Structure

```
├── agents/                        # Agent implementation code
│   ├── agent_config.py           # Configuration classes for the agent
│   ├── function_calling_agent_w_retriever_tool.py  # Agent implementation with retriever tool
│   └── generated_configs/        # Generated agent configuration files
├── tests/                        # Unit tests
├── utils/                        # Utility functions and helpers
│   ├── build_retriever_index.py  # Vector search index creation
│   ├── chunk_docs.py            # Document chunking utilities
│   ├── eval_set_utilities.py    # Evaluation set creation helpers
│   ├── file_loading.py          # File loading utilities
│   └── typed_dicts_to_spark_schema.py  # Schema conversion utilities
├── validators/                   # Configuration validators
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- Databricks Runtime 14.0 or higher
- A Databricks workspace with access to:
  - [Mosaic AI Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/build-genai-apps.html)
  - [Mosaic AI Agent Evaluation](https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html)
  - [Vector Search](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html)
  - [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)

### Setup Steps

1. **Configure Global Settings** (00_global_config.py):
   - Set up Unity Catalog locations for your agent
   - Configure MLflow experiment tracking
   - Define evaluation settings

2. **Build Data Pipeline** (02_data_pipeline.py):
   - Load and parse your documents
   - Create chunks for vector search
   - Build the vector search index

3. **Create Agent** (03_agent_proof_of_concept.py):
   - Configure the agent with LLM and retriever settings
   - Deploy the agent to collect feedback

4. **Evaluate and Improve** (04_create_evaluation_set.py, 05_evaluate_poc_quality.py):
   - Create evaluation sets from feedback
   - Measure quality metrics
   - Identify and fix quality issues

## Key Components

### Agent Configuration

The agent is configured using the `AgentConfig` class in `agents/agent_config.py`. Key configuration includes:

- Retriever tool settings (vector search, chunk formatting)
- LLM configuration (model endpoint, system prompts)
- Input examples for testing

### Data Pipeline

The data pipeline handles:

- Document loading and parsing
- Text chunking with configurable strategies
- Vector index creation for retrieval

### Quality Evaluation

The evaluation framework provides:

- Feedback collection through the Review App
- Quality metrics computation
- Root cause analysis of issues
- Iterative quality improvements

## Usage Example

```python
# 1. Configure your agent
from agents.agent_config import AgentConfig, RetrieverToolConfig, LLMConfig

agent_config = AgentConfig(
    retriever_tool=RetrieverToolConfig(...),
    llm_config=LLMConfig(...),
    input_example={...}
)

# 2. Initialize and test the agent
from agents.function_calling_agent_w_retriever_tool import AgentWithRetriever

agent = AgentWithRetriever()
response = agent.predict(model_input={"messages": [{"role": "user", "content": "What is RAG?"}]})
```

## Contributing

1. Follow the [development setup](../dev/README.md) instructions
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Additional Resources

- [Databricks Generative AI Cookbook](https://ai-cookbook.io/)
- [Mosaic AI Documentation](https://docs.databricks.com/en/generative-ai/index.html)
- [Vector Search Documentation](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html) 