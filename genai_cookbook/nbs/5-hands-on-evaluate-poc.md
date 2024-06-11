## **Step 4:** Evaluate the POC's quality

```{image} ../images/5-hands-on/workflow_baseline.png
:align: center
```

<br/>

**Expected time:** 5 - 60 minutes

*Time varies based on the number of questions in your evaluation set.  For 100 questions, evaluation will take approximately 5 minutes.*

```{admonition} [Code Repository](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code)
:class: tip
You can find all of the sample code referenced throughout this section [here](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code).
```

### **Overview & expected outcome**

This step will use the evaluation set you just curated to evaluate your POC app and establish the baseline quality/cost/latency.  The evaluation results are used by the next step to root cause any quality issues.

Evaluation is done using [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) and looks comprehensively across all aspects of quality, cost, and latency outlined in the [metrics](./4-evaluation-metrics.md) section of this cookbook.  

The aggregated metrics and evaluation of each question in the evaluation set are logged to MLflow.  For more details, see the [evaluation outputs](https://docs.databricks.com/generative-ai/agent-evaluation/evaluate-agent.html#evaluation-outputs) documentation.

### **Requirements**

- Your Evaluation Set is available
- All requirements from previous steps

### **Instructions**

1. Open the `05_evaluate_poc_quality` Notebook within your chosen POC directory and press Run All.

2. Inspect the results of evaluation in the Notebook or using MLflow.

```{note}
If the results meet your requirements for quality, you can skip directly to the Deployment section.  Because the POC application is built on Databricks, it is ready to be deployed to a scalable, production-ready REST API.
```

> **Next step:** Using this baseline evaluation of the POC's quality, identify the [root causes](./5-hands-on-improve-quality.md) of any quality issues and iteratively fix those issues to improve the app.

