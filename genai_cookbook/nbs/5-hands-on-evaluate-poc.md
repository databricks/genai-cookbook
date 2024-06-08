## Evaluate the POC's quality

```{image} ../images/5-hands-on/11_img.png
:align: center
```

**Expected time:** 30-60 minutes

**Requirements:**

- eval set from previsou steo

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
