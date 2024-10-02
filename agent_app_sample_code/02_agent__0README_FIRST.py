# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## To create your Agent, first select the Agent design pattern and SDK you wish to use.
# MAGIC
# MAGIC #### TLDR:
# MAGIC If you aren't sure, use the `02_agent__function_calling_mlflow_sdk` Notebook, which implements a function-calling Agent with the MLflow Deployments SDK & Databricks Model Serving.  This pattern provides more flexibility to tune the quality of the Agent's responses and add additional tools later, such as querying an additional vector store or Delta Table.
# MAGIC
# MAGIC
# MAGIC #### This cookbook provides 2 Agent design patterns:
# MAGIC 1. **Function-calling Agent:** An Agent that uses the [function-calling pattern](https://docs.databricks.com/en/machine-learning/model-serving/function-calling.html) to decide if querying a Vector Search index is necessary to answer a user's query.  This Agent can be extended by adding additional tools, such as a 
# MAGIC 2. **RAG-only Agent:** An Agent that always queries a Vector Search index to answer a user's query.
# MAGIC
# MAGIC
# MAGIC #### This cookbook implements 2 SDKs:
# MAGIC 1. Python code + MLflow Deployments SDK.  This template uses the MLflow Deployments SDK to call to the Databricks Model Serving, implementing the Agent's logic in pure Python code.
# MAGIC     - `02_agent__function_calling_mlflow_sdk`
# MAGIC     - `02_agent__RAG_only_mlflow_sdk`
# MAGIC 2. LangChain SDK.  This template uses LangChain and the Databricks LangChain SDK to call to Databricks Model serving.
# MAGIC     - `02_agent__RAG_only_langchain_sdk`
