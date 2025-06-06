# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Create a custom scorer that wraps the guidelines judge
# MAGIC
# MAGIC This script contains the code from the following documentation page: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/meets-guidelines
# MAGIC
# MAGIC Please refer to the documentation page for more information and a step-by-step guide.


# COMMAND ----------

# MAGIC dbutils.library.restartPython()


# COMMAND ----------

from typing import Dict, Any
from typing import List, Dict
import json
import os
from openai import OpenAI
from mlflow.genai.judges import meets_guidelines
from mlflow.genai.scorers import scorer
import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 1: Create the sample app to evaluate


# COMMAND ----------

mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)

# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the guidelines scorers handle rude and verbose responses
FOLLOW_POLICIES = False

# This is a global variable that will be used to toggle the behavior of the customer support agent to see how the guidelines scorers handle rude and verbose responses
BE_RUDE_AND_VERBOSE = False

@mlflow.trace
def customer_support_agent(user_messages: List[Dict[str, str]], user_id: str):

    # 1. Fake policies to follow.
    @mlflow.trace
    def get_policies_for_user(user_id: str):
        if user_id == 1:
            return [
                "All returns must be processed within 30 days of purchase, with a valid receipt.",
            ]
        else:
            return [
                "All returns must be processed within 90 days of purchase, with a valid receipt.",
            ]

    policies_to_follow = get_policies_for_user(user_id)

    # 2. Prepare messages for the LLM
    # We will use this toggle later to see how the scorers handle rude and verbose responses
    system_prompt_postfix = (
        f"Follow the following policies: {policies_to_follow}.  Do not refer to the specific policies in your response.\n"
        if FOLLOW_POLICIES
        else ""
    )

    system_prompt_postfix = (
        f"{system_prompt_postfix}Be super rude and very verbose in your responses.\n"
        if BE_RUDE_AND_VERBOSE
        else system_prompt_postfix
    )
    messages_for_llm = [
        {
            "role": "system",
            "content": f"You are a helpful customer support agent.  {system_prompt_postfix}",
        },
        *user_messages,
    ]

    # 3. Call LLM to generate a response
    output = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",  # This example uses Databricks hosted Claude 3.7 Sonnet. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o, etc.
        messages=messages_for_llm,
    )

    return {
        "message": output.choices[0].message.content,
        "policies_followed": policies_to_follow,
    }

result = customer_support_agent(
    user_messages=[
        {"role": "user", "content": "How much does a microwave cost?"},
    ],
    user_id=1
)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 2: Define your evalation criteria and wrap as custom scorers


# COMMAND ----------

tone = "The response must maintain a courteous, respectful tone throughout.  It must show empathy for customer concerns."
structure = "The response must use clear, concise language and structures responses logically.  It must avoids jargon or explains technical terms when used."
banned_topics = "If the request is a question about product pricing, the response must politely decline to answer and refer the user to the pricing page."
relevance = "The response must be relevant to the user's request.  Only consider the relevance and nothing else. If the request is not clear, the response must ask for more information."
# Note in this guideline how we refer to `provided_policies` - we will make the meets_guidelines LLM judge aware of this data.
follows_policies_guideline = "If the provided_policies is relevant to the request and response, the response must adhere to the provided_policies."

# Define a custom scorer that wraps the guidelines LLM judge to check if the response follows the policies
@scorer
def follows_policies(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    # we directly return the Feedback object from the guidelines LLM judge, but we could have post-processed it before returning it.
    return meets_guidelines(
        name="follows_policies",
        guidelines=follows_policies_guideline,
        context={
            # Here we make meets_guidelines aware of
            "provided_policies": outputs["policies_followed"],
            "response": outputs["message"],
            "request": json.dumps(inputs["user_messages"]),
        },
    )


# Define a custom scorer that wraps the guidelines LLM judge to pass the custom keys from the inputs/outputs to the guidelines LLM judge
@scorer
def check_guidelines(inputs: Dict[Any, Any], outputs: Dict[Any, Any]):
    feedbacks = []

    request = json.dumps(inputs["user_messages"])
    response = outputs["message"]

    feedbacks.append(
        meets_guidelines(
            name="tone",
            guidelines=tone,
            # Note: While we used request and response as keys, we could have used any key as long as our guideline referred to that key by name (e.g., if we had used output instead of response, we would have changed our guideline to be "The output must be polite")
            context={"response": response},
        )
    )

    feedbacks.append(
        meets_guidelines(
            name="structure",
            guidelines=structure,
            context={"response": response},
        )
    )

    feedbacks.append(
        meets_guidelines(
            name="banned_topics",
            guidelines=banned_topics,
            context={"request": request, "response": response},
        )
    )

    feedbacks.append(
        meets_guidelines(
            name="relevance",
            guidelines=relevance,
            context={"request": request, "response": response},
        )
    )

    # A scorer can return a list of Feedback objects OR a single Feedback object.
    return feedbacks

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 3: Create a sample evaluation dataset


# COMMAND ----------

eval_dataset = [
    {
        "inputs": {
            # Note that these keys match the **kwargs of our application.
            "user_messages": [
                {"role": "user", "content": "How much does a microwave cost?"},
            ],
            "user_id": 3,
        },
    },
    {
        "inputs": {
            "user_messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
            "user_id": 1,  # the bot should say no if the policies are followed for this user
        },
    },
    {
        "inputs": {
            "user_messages": [
                {
                    "role": "user",
                    "content": "Can I return the microwave I bought 2 months ago?",
                },
            ],
            "user_id": 2,  # the bot should say yes if the policies are followed for this user
        },
    },
    {
        "inputs": {
            "user_messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "Website"},
            ],
            "user_id": 3,
        },
    },
    {
        "inputs": {
            "user_messages": [
                {
                    "role": "user",
                    "content": "I'm having trouble with my account.  I can't log in.",
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear that you're having trouble with your account.  Are you using our website or mobile app?",
                },
                {"role": "user", "content": "JUST FIX IT FOR ME"},
            ],
            "user_id": 1,
        },
    },
]

print(eval_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 4: Evaluate your app using the guidelines


# COMMAND ----------

# Now, let's evaluate the app's responses against the guidelines when it is NOT rude and verbose and DOES follow policies
BE_RUDE_AND_VERBOSE = False
FOLLOW_POLICIES = True

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[follows_policies, check_guidelines],
)


# Now, let's evaluate the app's responses against the guidelines when it IS rude and verbose and does NOT follow policies
BE_RUDE_AND_VERBOSE = True
FOLLOW_POLICIES = False

mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=customer_support_agent,
    scorers=[follows_policies, check_guidelines],
)
