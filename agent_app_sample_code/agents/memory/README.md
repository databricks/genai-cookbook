# Persistent Memory for Databricks Agents

This module adds persistent memory to Databricks agents using [soul.py](https://github.com/menonpg/soul.py)'s hybrid RAG+RLM architecture.

## Why Memory?

Standard LLM agents are stateless — each query starts fresh. Memory-enabled agents:

- Remember decisions from previous sessions
- Learn patterns from accumulated interactions  
- Build context over time without manual prompt engineering

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Query                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Memory Router                                    │
│              (classifies query complexity)                           │
│                                                                      │
│    FOCUSED (~90%)              EXHAUSTIVE (~10%)                    │
│    ↓                           ↓                                    │
│    RAG (vector search)         RLM (recursive synthesis)            │
│    Sub-second latency          Thorough reasoning                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Unity Catalog                                    │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│    │  Identity   │    │ Memory Log  │    │  Decisions  │           │
│    │  (SOUL.md)  │    │ (MEMORY.md) │    │ (explicit)  │           │
│    └─────────────┘    └─────────────┘    └─────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create Memory Tables

```sql
-- Run in your Databricks SQL warehouse
CREATE SCHEMA IF NOT EXISTS catalog.agent_memory;

-- Identity table (agent personality/instructions)
CREATE TABLE IF NOT EXISTS catalog.agent_memory.identity (
    agent_id STRING NOT NULL,
    soul_content TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT current_timestamp(),
    version INT DEFAULT 1
) USING DELTA;

-- Memory log (conversation history)
CREATE TABLE IF NOT EXISTS catalog.agent_memory.memory_log (
    agent_id STRING NOT NULL,
    timestamp TIMESTAMP DEFAULT current_timestamp(),
    role STRING NOT NULL,
    content TEXT NOT NULL,
    metadata MAP<STRING, STRING>
) USING DELTA
PARTITIONED BY (agent_id);

-- Decisions (explicit choices with rationale)
CREATE TABLE IF NOT EXISTS catalog.agent_memory.decisions (
    agent_id STRING NOT NULL,
    decision TEXT NOT NULL,
    context TEXT,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp(),
    confidence DOUBLE DEFAULT 1.0
) USING DELTA;
```

### 2. Install Dependencies

```bash
pip install soul-agent[anthropic]
```

### 3. Use the Memory-Enabled Agent

```python
from agents.memory.delta_backend import DeltaMemoryBackend
from agents.memory.memory_agent import MemoryEnabledAgent

# Initialize with your catalog
backend = DeltaMemoryBackend(
    catalog="your_catalog",
    schema="agent_memory",
    agent_id="my-agent"
)

agent = MemoryEnabledAgent(backend=backend)

# The agent now remembers across sessions
response = agent.ask("What patterns have we seen in customer complaints?")
print(response["answer"])
print(f"Route: {response['route']}")  # 'RAG' or 'RLM'
```

## Integration with ai_query()

Expose memory operations via SQL:

```sql
-- Ask with memory context
SELECT ai_query(
    'memory-agent-endpoint',
    '{"operation": "ask", "query": "What did we decide about escalation policy?"}'
) AS response;

-- Store a memory
SELECT ai_query(
    'memory-agent-endpoint',
    '{"operation": "remember", "query": "Customer prefers email contact"}'
);

-- Log a decision
SELECT ai_query(
    'memory-agent-endpoint',
    '{
        "operation": "decide",
        "decision": "Route VIP tickets to senior support",
        "rationale": "VIP customers have SLA requirements"
    }'
);
```

## Testing

### Local Unit Tests

```bash
# Install test dependencies
pip install pytest

# Run unit tests (mocked Spark)
pytest agents/memory/tests/ -v
```

### Databricks Integration Test

Run in a Databricks notebook:

```python
from agents.memory.tests.test_memory import test_integration_databricks
test_integration_databricks()
```

This will:
1. Create a test schema with memory tables
2. Test all backend operations (log, retrieve, search)
3. Clean up after itself

### Manual Verification

```python
# In a Databricks notebook
from agents.memory.delta_backend import DeltaMemoryBackend

backend = DeltaMemoryBackend(
    catalog="your_catalog",
    schema="agent_memory",
    agent_id="test"
)

# Store something
backend.log_memory("Test entry", role="user")

# Retrieve it
print(backend.get_memory(limit=10))

# Log a decision
backend.log_decision(
    decision="Test decision",
    context="Testing",
    rationale="Verification"
)

# Check decisions
print(backend.get_decisions())
```

## Files

- `delta_backend.py` — Unity Catalog storage backend for soul.py
- `memory_agent.py` — Memory-enabled agent wrapper
- `endpoint_model.py` — MLflow model for Model Serving deployment
- `tests/test_memory.py` — Unit and integration tests

## How It Works

soul.py uses two retrieval strategies:

1. **RAG (Retrieval-Augmented Generation)** — Vector search over memory, sub-second latency. Used for direct recall questions.

2. **RLM (Recursive Language Model)** — Synthesizes across all memories recursively. Used for complex reasoning that requires connecting multiple pieces.

A lightweight router (~100 tokens) classifies each query and routes automatically.

## References

- [soul.py Documentation](https://blog.themenonlab.com/blog/soul-py-persistent-memory-llm-agents/)
- [RAG + RLM Architecture](https://blog.themenonlab.com/blog/soul-py-v2-rag-rlm-hybrid/)
- [LoCoMo Benchmark Results](https://menonpg.github.io/soul-benchmarks/)
- [ai_query() Reference](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query)
