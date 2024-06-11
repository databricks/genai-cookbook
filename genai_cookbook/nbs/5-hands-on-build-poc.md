## **Step 2:** Deploy POC to collect stakeholder feedback

```{image} ../images/5-hands-on/workflow_poc.png
:align: center
```
<br/>

**Expected time:** 30-60 minutes

**Requirements**
1. Completed [start here](./6-implement-overview.md) steps
2. Data from your [requirements](/nbs/5-hands-on-requirements.md#requirements-questions) is available in your [Lakehouse](https://www.databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html) inside a Unity Catalog [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) <!-- or [Delta Table](https://docs.databricks.com/en/delta/index.html)-->

```{admonition} [Code Repository](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code)
:class: tip
You can find all of the sample code referenced throughout this section [here](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code).
```

**Expected outcome**

At the end of this step, you will have deployed the [Agent Evaluation Review App](https://docs.databricks.com/generative-ai/agent-evaluation/human-evaluation.html) which allows your stakeholders to test and provide feedback on your POC. Detailed logs from your stakeholder's usage and their feedback will flow to Delta Tables in your Lakehouse.

```{image} ../images/5-hands-on/review_app2.gif
:align: center
```
<br/>

**Overview**

The first step in evaluation-driven development is to build a proof of concept (POC). A POC offers several benefits:

1. Provides a directional view on the feasibility of your use case with RAG
2. Allows collecting initial feedback from stakeholders, which in turn enables you to create the first version of your Evaluation Set
3. Establishes a baseline measurement of quality to start to iterate from

Databricks recommends building your POC using the simplest RAG architecture and our recommended defaults for each knob/parameter.  


```{note}
**Why start from a simple POC?** There are hundreds of possible combinations of knobs you can tune within your RAG application. You can easily spend weeks tuning these knobs, but if you do so before you can systematically evaluate your RAG, you'll end up in what we call the POC doom loop—iterating on settings, but with no way to objectively know if you made an improvement—all while your stakeholders sit around impatiently waiting.
```

The POC template in this cookbook are designed with quality iteration in mind.  That is, they are parameterized with the knobs that our research has shown are important to tune in order to improve RAG quality.  Said differently, these templates are not "3 lines of code that magically make a RAG"—rather, they are a well-structured RAG application that can be tuned for quality in the following steps of an evaluation-driven development workflow.  

This enables you to quickly deploy a POC, but transition quickly to quality iteration without needing to rewrite your code.

Below is the technical architecture of the POC application:

```{image} ../images/5-hands-on/5_img.png
:align: center
```

```{note}
By default, the POC uses the open source models available on [Mosaic AI Foundation Model Serving](https://www.databricks.com/product/pricing/foundation-model-serving). However, because the POC uses Mosaic AI Model Serving, which supports *any foundation model*, using a different model is easy - simply configure that model in Model Serving and then replace the `embedding_endpoint_name` and `llm_endpoint_name` in the `00_config` Notebook.

- Follow [these steps](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html) for other open source models available in the Databricks Marketplace
- Follow this [notebook](REPO_URL/helpers/Create_OpenAI_External_Model.py) or these [instructions](https://docs.databricks.com/en/generative-ai/external-models/index.html) for 3rd party models such as Azure OpenAI, OpenAI, Cohere, Anthropic, Google Gemini, etc.
```


**Instructions**



1. **Open the POC code folder within [`A_POC_app`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/A_POC_app) based on your type of data:**

   <br/>

   | File type                  | Source                 | POC application folder |
   |----------------------------|------------------------|------------------------|
   | PDF files                  | UC Volume              |   [`pdf_uc_volume`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/A_POC_app/pdf_uc_volume)                     |
   | Powerpoint files           | UC Volume              |        [`pptx_uc_volume`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/A_POC_app/pptx_uc_volume)                |
   | DOCX files                 | UC Volume              |        [`docx_uc_volume`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/A_POC_app/docx_uc_volume)                |
   | JSON files w/ text/markdown/HTML content & metadata | UC Volume  |              [`json_uc_volume`](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code/A_POC_app/html_uc_volume)          |  
   <!--| HTML content               | Delta Table            |                        |
   | Markdown or regular text   | Delta Table            |                        | -->

   If your data doesn't meet one of the above requirements, you can customize the parsing function (`parser_udf`) within `02_poc_data_pipeline` in the above POC directories to work with your file types.

   Inside the POC folder, you will see the following notebooks:

```{image} ../images/5-hands-on/6_img.png
:align: center
```

```{tip}
The notebooks referenced below are relative to the specific POC you've chosen. For example, if you see a reference to `00_config` and you've chosen `pdf_uc_volume`, you'll find the relevant `00_config` notebook at [`A_POC_app/pdf_uc_volume/00_config`](https://github.com/databricks/genai-cookbook/blob/main/rag_app_sample_code/A_POC_app/pdf_uc_volume/00_config.py).
```

<br/>

2. **Optionally, review the default parameters**

   Open the `00_config` Notebook within the POC directory you chose above to view the POC's applications default parameters for the data pipeline and RAG chain.


   ```{note}
   **Important:** our recommended default parameters are by no means perfect, nor are they intended to be. Rather, they are a place to start from - the next steps of our workflow guide you through iterating on these parameters.
   ```

3. **Validate the configuration**

   Run the `01_validate_config` to check that your configuration is valid and all resources are available. You will see an `rag_chain_config.yaml` file appear in your directory - we will use this in step 4 to deploy the application.

4. **Run the data pipeline**

   The POC data pipeline is a Databricks Notebook based on Apache Spark. Open the `02_poc_data_pipeline` Notebook and press Run All to execute the pipeline. The pipeline will:

   1. Load the raw documents from the UC Volume
   2. Parse each document, saving the results to a Delta Table
   3. Chunk each document, saving the results to a Delta Table
   4. Embed the documents and create a Vector Index using Mosaic AI Vector Search

   <br/>

   Metadata (output tables, configuration, etc) about the data pipeline are logged to MLflow:

   ```{image} ../images/5-hands-on/datapipelinemlflow.gif
   :align: center
   ```
   

   <br/>

   You can inspect the outputs by looking for links to the Delta Tables/Vector Indexes output near the bottom of the notebook:

      ```
      Vector index: https://<your-workspace-url>.databricks.com/explore/data/<uc-catalog>/<uc-schema>/<app-name>_poc_chunked_docs_gold_index

      Output tables:

      Bronze Delta Table w/ raw files: https://<your-workspace-url>.databricks.com/explore/data/<uc-catalog>/<uc-schema>/<app-name>__poc_raw_files_bronze
      Silver Delta Table w/ parsed files: https://<your-workspace-url>.databricks.com/explore/data/<uc-catalog>/<uc-schema>/<app-name>__poc_parsed_docs_silver
      Gold Delta Table w/ chunked files: https://<your-workspace-url>.databricks.com/explore/data/<uc-catalog>/<uc-schema>/<app-name>__poc_chunked_docs_gold
      ```

5. **Deploy the POC chain to the Review App**

   The default POC chain is a multi-turn conversation RAG chain built using LangChain.

   ```{tip}
   The POC Chain uses MLflow code-based logging. To understand more about code-based logging, visit the [docs](https://docs.databricks.com/generative-ai/create-log-agent.html#code-based-vs-serialization-based-logging).
   ```

   1. Open the `03_deploy_poc_to_review_app` Notebook

   2. Run each cell of the Notebook.

   3. You will see the MLflow Trace that shows you how the POC application works. Adjust the input question to one that is relevant to your use case, and re-run the cell to "vibe check" the application.

      ```{image} ../images/5-hands-on/mlflow_trace2.gif
      :align: center
      ```

   4. Modify the default instructions to be relevant to your use case.  These are displayed in the Review App.

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

      ```
      Review App URL: https://<your-workspace-url>.databricks.com/ml/review/<uc-catalog>.<uc-schema>.<uc-model-name>/<uc-model-version>
      ```

6. **Grant individual users permissions to access the Review App.**  

   You can grant access to non-Databricks users by following these [steps](https://docs.databricks.com/generative-ai/agent-evaluation/human-evaluation.html#set-up-sso-permissions-to-the-review-app-workspace).

7. **Test the Review App by asking a few questions yourself and providing feedback.**
   
   ```{note}
   MLflow Traces and the user's feedback from the Review App will appear in Delta Tables in the catalog/schema you have configured. Logs can take up to 2 hours to appear in these Delta Tables.
   ```

8. **Share the Review App with stakeholders**

   You can now share your POC RAG application with your stakeholders to get their feedback.

   ```{important}
   We suggest distributing your POC to at least 3 stakeholders and having them each ask 10 - 20 questions. It is important to have multiple stakeholders test your POC so you can have a diverse set of perspectives to include in your Evaluation Set.
   ```

