## Evaluate the POC's quality

```{image} ../images/5-hands-on/11_img.png
:align: center
```

**Expected time:** 30-60 minutes

**Requirements:**

- Stakeholders have used your POC and provided feedback
- All requirements from [POC step](#how-to-build-a-poc)
  - Data from your [requirements](#requirements-questions) is available in your [Lakehouse](https://www.databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html) inside a [Unity Catalog](https://www.databricks.com/product/unity-catalog) [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html) or [Delta Table](https://docs.databricks.com/en/delta/index.html)
  - Access to a [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) endpoint [[instructions](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html)]
  - Write access to Unity Catalog schema
  - A single-user cluster with DBR 14.3+

Now that your stakeholders have used your POC, we can use their feedback to measure the POC's quality and establish a baseline.

#### 1. ETL the logs to an Evaluation Set & run evaluation

1. Open the `04_evaluate_poc_quality` Notebook.

2. Adjust the configuration at the top to point to your Review App's logs.

3. Run the cell to create an initial Evaluation Set that includes
   - 3 types of logs
     1. Requests with a 👍 :
        - `request`: As entered by the user
        - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
     2. Requests with a 👎 :
        - `request`: As entered by the user
        - `expected_response`: If the user edited the response, that is used, otherwise, null.
     3. Requests without any feedback
        - `request`: As entered by the user
   - Across all types of requests, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.

> note: Databricks recommends that your Evaluation Set contains at least 30 questions to get started.

4. Inspect the Evaluation Set to understand the data that is included. You need to validate that your Evaluation Set contains a representative and challenging set of questions.

5. Optionally, save your evaluation set to a Delta Table for later use

6. Evaluate the POC with Quality Lab's LLM Judge-based evaluation. Open MLflow to view the results.

```{image} ../images/5-hands-on/12_img.png
:align: center
```

```{image} ../images/5-hands-on/13_img.png
:align: center
```

#### 2. Review evaluation results

1. Now, let's open MLflow to inspect the results.

2. In the Run tab, we can see each of the computed metrics. Refer to [metrics overview] section for an explanation of what each metric tells you about your application.

3. In the Evaluation tab, we can inspect the questions, RAG application's outputs, and each of the LLM judge's assessments.

Now that you have a baseline understanding of the POC's quality, we can shift focus to identifying the root causes of any quality issues and iteratively improving the app.

It is worth noting: if the results meet your requirements for quality, you can skip directly to the Deployment section.
