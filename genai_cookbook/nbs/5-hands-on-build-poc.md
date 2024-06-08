## Deploy POC to collect stakeholder feedback

```{image} ../images/5-hands-on/3_img.png
:align: center
```

The first step in evaluation-driven development is to build a proof of concept (POC). A POC offers several benefits:

1. Provides a directional view on the feasibility of your use case with RAG
2. Allows collecting initial feedback from stakeholders, which in turn enables you to create the first version of your Evaluation Set
3. Establishes a baseline measurement of quality to start to iterate from

Databricks recommends building your POC using the simplest RAG chain architecture and our recommended defaults for each knob/parameter.  

> *!! Important: our recommended default parameters are by no means perfect, nor are they intended to be. Rather, they are a place to start from - the next steps of our workflow guide you through iterating on these parameters.*
>
> *Why start from a simple POC? There are hundreds of possible combinations of knobs you can tune within your RAG application. You can easily spend weeks tuning these knobs, but if you do so before you can systematically evaluate your RAG, you'll end up in what we call the POC doom loop—iterating on settings, but with no way to objectively know if you made an improvement—all while your stakeholders sit around impatiently waiting.*

The POC templates in this guide are designed with quality iteration in mind—that is, they are parameterized with the knobs that our research has shown are most important to tune in order to improve RAG quality. Each knob has a smart default.

Said differently, these templates are not "3 lines of code that magically make a RAG"—rather, they are a well-structured RAG application that can be tuned for quality in the following steps of an evaluation-driven development workflow.  

This enables you to quickly deploy a POC, but transition quickly to quality iteration without needing to rewrite your code.

### How to build a POC

**Expected time:** 30-60 minutes

**Requirements:**

- Data from your [requirements](/nbs/5-hands-on-requirements.md#requirements-questions) is available in your [Lakehouse](https://www.databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html) inside a [Unity Catalog](https://www.databricks.com/product/unity-catalog) [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) or [Delta Table](https://docs.databricks.com/en/delta/index.html)
- Access to a [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) endpoint [[instructions](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html)]
- Write access to Unity Catalog schema
- A single-user cluster with DBR 14.3+

At the end of this step, you will have deployed the Quality Lab Review App which allows your stakeholders to test and provide feedback on your POC. Detailed logs from your stakeholder's usage and their feedback will flow to Delta Tables in your Lakehouse.

```{image} ../images/5-hands-on/4_img.png
:align: center
```

Below is the technical architecture of the POC application.

```{image} ../images/5-hands-on/5_img.png
:align: center
```

By default, the POC uses the open source models available on [Mosaic AI Foundation Model Serving](https://www.databricks.com/product/pricing/foundation-model-serving). However, because the POC uses Mosaic AI Model Serving, which supports *any foundation model*, using a different model is easy - simply configure that model in Model Serving and then replace the embedding_endpoint_name and llm_endpoint_name parameters in the POC code.

- [follow these steps for other open source models in the marketplace e.g., PT]
- [follow these steps for models such as Azure OpenAI, OpenAI, Cohere, Anthropic, Google Gemini, etc e.g., external models]

#### 1. Import the sample code.

To get started, [import this Git Repository to your Databricks Workspace](https://docs.databricks.com/en/repos/index.html). This repository contains the entire set of sample code. Based on your data, select one of the following folders that contains the POC application code.

| File type                  | Source                 | POC application folder |
|----------------------------|------------------------|------------------------|
| PDF files                  | UC Volume              |                        |
| JSON files w/ HTML content & metadata | UC Volume  |                        |  
| Powerpoint files           | UC Volume              |                        |
| DOCX files                 | UC Volume              |                        |
| HTML content               | Delta Table            |                        |
| Markdown or regular text   | Delta Table            |                        |

If you don't have any data ready, and just want to follow along using the Databricks Customer Support Bot example, you can use this pipeline which uses a Delta Table of the Databricks Docs stored as HTML.

If your data doesn't meet one of the above requirements, [insert instructions on how to customize].

Once you have imported the code, you will have the following notebooks:

```{image} ../images/5-hands-on/6_img.png
:align: center
```

#### 2. Configure your application

Follow the instructions in the `00_config` Notebook to configure the following settings:

1. `RAG_APP_NAME`: The name of the RAG application. This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes

2. `UC_CATALOG` & `UC_SCHEMA`: [Create Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored

3. `UC_MODEL_NAME`: Unity Catalog location to log and store the chain's model

4. `VECTOR_SEARCH_ENDPOINT`: [Create Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to host the resulting vector index

5. `SOURCE_PATH`: [Create Volumes](https://docs.databricks.com/en/connect/unity-catalog/volumes.html#create-and-work-with-volumes) for source documents as `SOURCE_PATH`

6. `MLFLOW_EXPERIMENT_NAME`: MLflow Experiment to use for this application. Using the same experiment allows you to track runs across Notebooks and store a single history for your application.

Run the `00_validate_config` to check that your configuration is valid and all resources are available. You will see an `rag_chain_config.yaml` file appear in your directory - we will do this in step 4 to deploy the application.

#### 3. Prepare your data.

The POC data pipeline is a Databricks Notebook based on Apache Spark that provides a default implementation of the parameters outlined below.  

To run this pipeline and generate your initial Vector Index:

1. Open the `02_poc_data_pipeline` Notebook and connect it to your single-user cluster

2. Press Run All to execute the data pipeline

3. In the last cell of the notebook, you can see the resulting Delta Tables and Vector Index.

```{image} ../images/5-hands-on/7_img.png
:align: center
```

Parameters and their default values that are configured in `00_config`.

| Knob                                     | Description                                                                                                                               | Default value                                                                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [Parsing strategy](/nbs/3-deep-dive-data-pipeline.md#parsing)             | Extracting relevant information from the raw data using appropriate parsing techniques                                                     | Varies based on document type, but generally an open source parsing library                                                                  |
| [Chunking strategy](/nbs/3-deep-dive-data-pipeline.md#chunking)           | Breaking down the parsed data into smaller, manageable chunks for efficient retrieval                                                     | Token Text Splitter, which splits text along using a chunk size of 4000 tokens and a stride of 500 tokens.                                    |
| [Embedding model](/nbs/3-deep-dive-data-pipeline.md#embedding-model)      | Converting the chunked text data into a numerical vector representation that captures its semantic meaning                                 | GTE-Large-v1.5 on the Databricks FMAPI pay-per-token                                                                                         |

#### 4. Deploy the POC chain to the Quality Lab Review App

The POC chain is a RAG chain that provides a default implementation of the parameters outlined below.

> Note: The POC Chain uses MLflow code-based logging. To understand more about code-based logging, [link to docs].

1. Open the `03_deploy_poc_to_review_app` Notebook

2. Run each cell of the Notebook.

3. You will see the MLflow Trace that shows you how the POC application works. Adjust the input question to one that is relevant to your use case, and re-run the cell to "vibe check" the application.

```{image} ../images/5-hands-on/8_img.png
:align: center
```

4. Modify the default instructions to be relevant to your use case.

```python
   instructions_to_reviewer = f"""## Instructions for Testing the {RAG_APP_NAME}'s Initial Proof of Concept (PoC)

   Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

   1. **Variety of Questions**:
      - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

   2. **Feedback on Answers**:
      - After asking each question, use the feedback widgets provided to review the answer given by the application.
      - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

   3. **Review of Returned Documents**:
      - Carefully review each document that the system returns in response to your question.
      - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

   Thank you for your time and effort in testing {RAG_APP_NAME}. Your contributions are essential to delivering a high-quality product to our end users."""

   print(instructions_to_reviewer)
```


5. Run the deployment cell to get a link to the Review App.

```{image} ../images/5-hands-on/9_img.png
:align: center
```

6. Grant individual users permissions to access the Review App.

```{image} ../images/5-hands-on/10_img.png
:align: center
```

7. Test the Review App by asking a few questions yourself and providing feedback.
   - You can view the data in Delta Tables. Note that results can take up to 2 hours to appear in the Delta Tables.

Parameters and their default values configured in 00_config:

| Knob | Description | Default value |
|------|-------------|---------------|
| [Query understanding](/nbs/3-deep-dive-chain.md#query-understanding) | Analyzing and transforming user queries to better represent intent and extract relevant information, such as filters or keywords, to improve the retrieval process. | None, the provided query is directly embedded. |
| [Retrieval](/nbs/3-deep-dive-chain.md#retrieval) | Finding the most relevant chunks of information given a retrieval query. In the unstructured data case, this typically involves one or a combination of semantic or keyword-based search. | Semantic search with K = 5 chunks retrieved |
| [Prompt augmentation](/nbs/3-deep-dive-chain.md#prompt-augmentation) | Combining a user query with retrieved information and instructions to guide the LLM towards generating high-quality responses. | A simple RAG prompt template |
| [LLM](/nbs/3-deep-dive-chain.md#llm) | Selecting the most appropriate model (and model parameters) for your application to optimize/balance performance, latency, and cost. | Databricks-dbrx-instruct hosted using Databricks FMAPI pay-per-token |
| [Post processing & guardrails](/nbs/3-deep-dive-chain.md#post-processing-guardrails) | Applying additional processing steps and safety measures to ensure the LLM-generated responses are on-topic, factually consistent, and adhere to specific guidelines or constraints. | None |

#### 5. Share the Review App with stakeholders

You can now share your POC RAG application with your stakeholders to get their feedback.

We suggest distributing your POC to at least 3 stakeholders and having them each ask 10 - 20 questions. It is important to have multiple stakeholders test your POC so you can have a diverse set of perspectives to include in your Evaluation Set.

