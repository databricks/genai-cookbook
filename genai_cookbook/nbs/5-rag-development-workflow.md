## Evaluation-driven development workflow


<!-- ## Evaluation-driven development -->
This section walks you through Databricks recommended development workflow for building, testing, and deploying a high-quality RAG application: **evaluation-driven development**. This workflow is based on the Mosaic Research team's recommended best practices for building and evaluating high quality RAG applications. If quality is important to your business, Databricks recommends following an evaluation-driven workflow:

1. Define the requirements
2. Collect stakeholder feedback on a rapid proof of concept (POC)
3. Evaluate the POC's quality
4. Iteratively diagnose and fix quality issues
5. Deploy to production
6. Monitor in production 

```{image} ../images/5-hands-on/workflow.png
:align: center
```
<br/>

The [implement](./6-implement-overview) section of this cookbook provides a guided implementation of this workflow with sample code.

There are two core concepts in **evaluation-driven development:**

1. [**Metrics:**](./4-evaluation-metrics.md) Defining what high-quality means

   *Similar to how you set business goals each year, you need to define what high-quality means for your use case.* *Databricks' Quality Lab provides a suggested set of* *N metrics to use, the most important of which is answer accuracy or correctness - is the RAG application providing the right answer?*

2. [**Evaluation set:**](./4-evaluation-eval-sets.md) Objectively measuring the metrics

   *To objectively measure quality, you need an evaluation set, which contains questions with known-good answers validated by humans. While this may seem scary at first - you probably don't have an evaluation set sitting ready to go - this guide walks you through the process of developing and iteratively refining this evaluation set.*

Anchoring against metrics and an evaluation set provides the following benefits:

1. You can iteratively and confidently refine your application's quality during development - no more guessing if a change resulted in an improvement.

2. Getting alignment with business stakeholders on the readiness of the application for production becomes more straightforward when you can confidently state, *"we know our application answers the most critical questions to our business correctly and doesn't hallucinate."*