#!/usr/bin/env python3
"""
# 1. Use the prebuilt guidelines scorer

This script contains the code from the following documentation page: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/meets-guidelines

Please refer to the documentation page for more information and a step-by-step guide."""


from typing import List, Dict
from typing import List, Dict, Any
import os

from openai import OpenAI

from mlflow.genai.scorers import Guidelines
import mlflow


print("=" * 80)
print("Guide: 1. Use the prebuilt guidelines scorer")
print("=" * 80)


# ================================================================================
# Step 1: Create the sample app to evaluate
# ================================================================================

print("\n" + "=" * 60)
print("Testing: Step 1: Create the sample app to evaluate")
print("=" * 60)

mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)

# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the guidelines scorers handle rude and verbose responses
BE_RUDE_AND_VERBOSE = False

@mlflow.trace
def customer_support_agent(messages: List[Dict[str, str]]):

    # 1. Prepare messages for the LLM
    system_prompt_postfix = (
        "Be super rude and very verbose in your responses."
        if BE_RUDE_AND_VERBOSE
        else ""
    )
    messages_for_llm = [
        {
            "role": "system",
            "content": f"You are a helpful customer support agent.  {system_prompt_postfix}",
        },
        *messages,
    ]

    # 2. Call LLM to generate a response
    return client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",  # This example uses Databricks hosted Claude 3.7 Sonnet. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        messages=messages_for_llm,
    )

result = customer_support_agent(
    messages=[
        {"role": "user", "content": "How much does a microwave cost?"},
    ]
)
print(result)


# ================================================================================
# Step 2: Define your evaluation criteria
# ================================================================================

print("\n" + "=" * 60)
print("Testing: Step 2: Define your evaluation criteria")
print("=" * 60)

tone = "The response must maintain a courteous, respectful tone throughout.  It must show empathy for customer concerns."
structure = "The response must use clear, concise language and structures responses logically.  It must avoids jargon or explains technical terms when used."
banned_topics = "If the request is a question about product pricing, the response must politely decline to answer and refer the user to the pricing page."
relevance = "The response must be relevant to the user's request.  Only consider the relevance and nothing else. If the request is not clear, the response must ask for more information."


# ================================================================================
# Step 3: Create a sample evaluation dataset
# ================================================================================

print("\n" + "=" * 60)
print("Testing: Step 3: Create a sample evaluation dataset")
print("=" * 60)

eval_dataset = [
    {
        "inputs": {
            "messages": [
                {"role": "user", "content": "How much does a microwave cost?"},
            ]
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "Website"},
            ]
        },
    },
    {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "JUST FIX IT FOR ME"},
            ]
        },
    },
]
print(eval_dataset)


# ================================================================================
# Step 4: Evaluate your app using the custom scorers
# ================================================================================

print("\n" + "=" * 60)
print("Testing: Step 4: Evaluate your app using the custom scorers")
print("=" * 60)

# First, let's evaluate the app's responses against the guidelines when it is not rude and verbose
BE_RUDE_AND_VERBOSE = False

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[
        Guidelines(name="tone", guidelines=tone),
        Guidelines(name="structure", guidelines=structure),
        Guidelines(name="banned_topics", guidelines=banned_topics),
        Guidelines(name="relevance", guidelines=relevance),
    ],
)


# Next, let's evaluate the app's responses against the guidelines when it IS rude and verbose
BE_RUDE_AND_VERBOSE = True

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[
        Guidelines(name="tone", guidelines=tone),
        Guidelines(name="structure", guidelines=structure),
        Guidelines(name="banned_topics", guidelines=banned_topics),
        Guidelines(name="relevance", guidelines=relevance),
    ],
)
