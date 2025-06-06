# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluating a version of your app
# MAGIC
# MAGIC This script contains the code from the following documentation page: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/evaluate-app
# MAGIC
# MAGIC Please refer to the documentation page for more information and a step-by-step guide.


# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0" openai "databricks-connect>=16.1"
# MAGIC dbutils.library.restartPython()


# COMMAND ----------

from typing import List, Dict
import os
import time
from openai import OpenAI
from databricks.connect import DatabricksSession
from mlflow.entities import Document
from mlflow.genai.scorers import ( RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines, )
import mlflow
import mlflow.genai.datasets

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 1: Create your application


# COMMAND ----------

# Enable automatic tracing for OpenAI calls
mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)

# Simulated CRM database
CRM_DATA = {
    "Acme Corp": {
        "contact_name": "Alice Chen",
        "recent_meeting": "Product demo on Monday, very interested in enterprise features. They asked about: advanced analytics, real-time dashboards, API integrations, custom reporting, multi-user support, SSO authentication, data export capabilities, and pricing for 500+ users",
        "support_tickets": ["Ticket #123: API latency issue (resolved last week)", "Ticket #124: Feature request for bulk import", "Ticket #125: Question about GDPR compliance"],
        "account_manager": "Sarah Johnson"
    },
    "TechStart": {
        "contact_name": "Bob Martinez", 
        "recent_meeting": "Initial sales call last Thursday, requested pricing",
        "support_tickets": ["Ticket #456: Login issues (open - critical)", "Ticket #457: Performance degradation reported", "Ticket #458: Integration failing with their CRM"],
        "account_manager": "Mike Thompson"
    },
    "Global Retail": {
        "contact_name": "Carol Wang",
        "recent_meeting": "Quarterly review yesterday, happy with platform performance",
        "support_tickets": [],
        "account_manager": "Sarah Johnson"
    }
}

# Use a retriever span to enable MLflow's predefined RetrievalGroundedness scorer to work
@mlflow.trace(span_type="RETRIEVER")
def retrieve_customer_info(customer_name: str) -> List[Document]:
    """Retrieve customer information from CRM database"""
    if customer_name in CRM_DATA:
        data = CRM_DATA[customer_name]
        return [
            Document(
                id=f"{customer_name}_meeting",
                page_content=f"Recent meeting: {data['recent_meeting']}",
                metadata={"type": "meeting_notes"}
            ),
            Document(
                id=f"{customer_name}_tickets",
                page_content=f"Support tickets: {', '.join(data['support_tickets']) if data['support_tickets'] else 'No open tickets'}",
                metadata={"type": "support_status"}
            ),
            Document(
                id=f"{customer_name}_contact",
                page_content=f"Contact: {data['contact_name']}, Account Manager: {data['account_manager']}",
                metadata={"type": "contact_info"}
            )
        ]
    return []

@mlflow.trace
def generate_sales_email(customer_name: str, user_instructions: str) -> Dict[str, str]:
    """Generate personalized sales email based on customer data & a sale's rep's instructions."""
    # Retrieve customer information
    customer_docs = retrieve_customer_info(customer_name)
    
    if not customer_docs:
        return {"error": f"No customer data found for {customer_name}"}
    
    # Combine retrieved context
    context = "\n".join([doc.page_content for doc in customer_docs])
    
    # Generate email using retrieved context
    prompt = f"""You are a sales representative. Based on the customer information below, 
    write a brief follow-up email that addresses their request.
    
    Customer Information:
    {context}
    
    User instructions: {user_instructions}
    
    Keep the email concise and personalized."""
    
    response = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=[
            {"role": "system", "content": "You are a helpful sales assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    
    return {"email": response.choices[0].message.content}

# Test the application
result = generate_sales_email("Acme Corp", "Follow up after product demo")
print(result["email"])

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 2: Simulate production traffic


# COMMAND ----------

# Simulate beta testing traffic with scenarios designed to fail guidelines
test_requests = [
    {"customer_name": "Acme Corp", "user_instructions": "Follow up after product demo"},
    {"customer_name": "TechStart", "user_instructions": "Check on support ticket status"},
    {"customer_name": "Global Retail", "user_instructions": "Send quarterly review summary"},
    {"customer_name": "Acme Corp", "user_instructions": "Write a very detailed email explaining all our product features, pricing tiers, implementation timeline, and support options"},  # Will likely fail conciseness guideline
    {"customer_name": "TechStart", "user_instructions": "Send an enthusiastic thank you for their business!"},  # Will likely include exclamation marks
    {"customer_name": "Global Retail", "user_instructions": "Send a follow-up email"},  # May not mention contact name
    {"customer_name": "Acme Corp", "user_instructions": "Just check in to see how things are going"},  # Vague, won't include specific next steps
]

# Run requests and capture traces
print("Simulating production traffic...")
for req in test_requests:
    try:
        result = generate_sales_email(**req)
        print(f"✓ Generated email for {req['customer_name']}")
    except Exception as e:
        print(f"✗ Error for {req['customer_name']}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 3: Create evaluation dataset


# COMMAND ----------

# 0. If you are using a local development environment, connect to Serverless Spark which powers MLflow's evaluation dataset service

mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = (
        lambda: "databricks-uc"
    )
spark = DatabricksSession.builder.remote(serverless=True).getOrCreate()

# 1. Create an evaluation dataset

# Replace with a Unity Catalog schema where you have CREATE TABLE permission
uc_schema = "workspace.default"
# This table will be created in the above UC schema
evaluation_dataset_table_name = "email_generation_eval"

eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}",
)
print(f"Created evaluation dataset: {uc_schema}.{evaluation_dataset_table_name}")

# 2. Search for the simulated production traces from step 2: get traces from the last 20 minutes with our trace name.
ten_minutes_ago = int((time.time() - 10 * 60) * 1000)

traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {ten_minutes_ago} AND "
                 f"attributes.status = 'OK' AND "
                 f"tags.`mlflow.traceName` = 'generate_sales_email'",
    order_by=["attributes.timestamp_ms DESC"]
)

print(f"Found {len(traces)} successful traces from beta test")

# 3. Add the traces to the evaluation dataset
eval_dataset.insert(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# Preview the dataset
df = eval_dataset.to_df()
print(f"\nDataset preview:")
print(f"Total records: {len(df)}")
print("\nSample record:")
sample = df.iloc[0]
print(f"Inputs: {sample['inputs']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 4: Run evaluation with predefined scorers


# COMMAND ----------

# Run evaluation with predefined scorers
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=generate_sales_email,
    scorers=[
        RetrievalGroundedness(),  # Checks if email content is grounded in retrieved data
        Guidelines(
            name="follows_instructions",
            guidelines="The generated email must follow the user_instructions in the request.",
        ),
        # Add strict guidelines that will likely fail
        Guidelines(
            name="concise_communication",
            guidelines="The email MUST be concise and to the point. The email should communicate the key message efficiently without being overly brief or losing important context.",
        ),
        Guidelines(
            name="mentions_contact_name",
            guidelines="The email MUST explicitly mention the customer contact's first name (e.g., Alice, Bob, Carol) in the greeting. Generic greetings like 'Hello' or 'Dear Customer' are not acceptable.",
        ),
        Guidelines(
            name="professional_tone",
            guidelines="The email must be in a professional tone.",
        ),
        Guidelines(
            name="includes_next_steps",
            guidelines="The email MUST end with a specific, actionable next step that includes a concrete timeline.",
        ),
        RelevanceToQuery(),  # Checks if email addresses the user's request
        Safety(),  # Checks for harmful or inappropriate content
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC # SECTION: Step 5: View and interpret results


# COMMAND ----------

eval_traces = mlflow.search_traces(run_id=eval_results.run_id)

# eval_traces is a Pandas DataFrame that has the evaluated traces.  The column `assessments` includes each scorer's feedback.
print(eval_traces)
