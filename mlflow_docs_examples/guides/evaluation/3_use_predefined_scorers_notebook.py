# Databricks notebook source
# MAGIC %md
# MAGIC # Use predefined LLM scorers
# MAGIC
# MAGIC This script contains the code from the following documentation page: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/predefined-judge-scorers
# MAGIC
# MAGIC Please refer to the documentation page for more information and a step-by-step guide.


# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0" openai
# MAGIC dbutils.library.restartPython()


# COMMAND ----------

from typing import List
import os
from openai import OpenAI
from mlflow.entities import Document
from mlflow.genai.scorers import ( Correctness, Guidelines, RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, RetrievalSufficiency, Safety, )
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 1: Create a sample application to evaluate


# COMMAND ----------

mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)


# Retriever function called by the sample app
@mlflow.trace(span_type="RETRIEVER")
def retrieve_docs(query: str) -> List[Document]:
    return [
        Document(
            id="sql_doc_1",
            page_content="SELECT is a fundamental SQL command used to retrieve data from a database. You can specify columns and use a WHERE clause to filter results.",
            metadata={"doc_uri": "http://example.com/sql/select_statement"},
        ),
        Document(
            id="sql_doc_2",
            page_content="JOIN clauses in SQL are used to combine rows from two or more tables, based on a related column between them. Common types include INNER JOIN, LEFT JOIN, and RIGHT JOIN.",
            metadata={"doc_uri": "http://example.com/sql/join_clauses"},
        ),
        Document(
            id="sql_doc_3",
            page_content="Aggregate functions in SQL, such as COUNT(), SUM(), AVG(), MIN(), and MAX(), perform calculations on a set of values and return a single summary value.  The most common aggregate function in SQL is COUNT().",
            metadata={"doc_uri": "http://example.com/sql/aggregate_functions"},
        ),
    ]


# Sample app that we will evaluate
@mlflow.trace
def sample_app(query: str):
    # 1. Retrieve documents based on the query
    retrieved_documents = retrieve_docs(query=query)
    retrieved_docs_text = "\n".join([doc.page_content for doc in retrieved_documents])

    # 2. Prepare messages for the LLM
    messages_for_llm = [
        {
            "role": "system",
            # Fake prompt to show how the various scorers identify quality issues.
            "content": f"Answer the user's question based on the following retrieved context: {retrieved_docs_text}.  Do not mention the fact that provided context exists in your answer.  If the context is not relevant to the question, generate the best response you can.",
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    # 3. Call LLM to generate the response
    return client.chat.completions.create(
        # This example uses Databricks hosted Claude.  If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        model="databricks-claude-3-7-sonnet",
        messages=messages_for_llm,
    )
result = sample_app("what is select in sql?")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 2: Create a sample evaluation dataset


# COMMAND ----------

eval_dataset = [
    {
        "inputs": {"query": "What is the most common aggregate function in SQL?"},
        "expectations": {
            "expected_facts": ["Most common aggregate function in SQL is COUNT()."],
        },
    },
    {
        "inputs": {"query": "How do I use MLflow?"},
        "expectations": {
            "expected_facts": [
                "MLflow is a tool for managing and tracking machine learning experiments."
            ],
        },
    },
]
print(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 3: Run evaluation with predefined scorers


# COMMAND ----------

# Run predefined scorers that require ground truth
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=sample_app,
    scorers=[
        Correctness(),
        # RelevanceToQuery(),
        # RetrievalGroundedness(),
        # RetrievalRelevance(),
        RetrievalSufficiency(),
        # Safety(),
    ],
)


# Run predefined scorers that do NOT require ground truth
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=sample_app,
    scorers=[
        # Correctness(),
        RelevanceToQuery(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        # RetrievalSufficiency(),
        Safety(),
        Guidelines(name="does_not_mention", guidelines="The response not mention the fact that provided context exists.")
    ],
)
