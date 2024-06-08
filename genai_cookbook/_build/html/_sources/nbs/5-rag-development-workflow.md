## RAG development workflow


<!-- ## Evaluation-driven development -->
This section walks you through Databricks recommended development workflow for building, testing, and deploying a high-quality RAG application: **evaluation-driven development**. This workflow is based on the Mosaic Research team's best practices for building and evaluating high quality RAG applications. If quality is important to your business, Databricks recommends following an evaluation-driven workflow:

1. Define the requirements
2. Collect stakeholder feedback on a rapid proof of concept (POC)
3. Evaluate the POC's quality
4. Iteratively diagnose and fix quality issues
5. Deploy to production
6. Monitor in production

```{image} ../images/5-hands-on/1_img.png
:align: center
```

Mapping to this workflow, this section provides ready-to-run sample code for every step and every suggestion to improve quality.

Throughout, we will demonstrate evaluation-driven development using one of Databricks' internal use generative AI cases: using a RAG bot to help answer customer support questions in order to [1] reduce support costs [2] improve the customer experience.

There are two core concepts in **evaluation-driven development:**

1. **Metrics:** Defining high-quality

   *Similar to how you set business goals each year, you need to define what high-quality means for your use case.* *Databricks' Quality Lab provides a suggested set of* *N metrics to use, the most important of which is answer accuracy or correctness - is the RAG application providing the right answer?*

2. **Evaluation:** Objectively measuring the metrics

   *To objectively measure quality, you need an evaluation set, which contains questions with known-good answers validated by humans. While this may seem scary at first - you probably don't have an evaluation set sitting ready to go - this guide walks you through the process of developing and iteratively refining this evaluation set.*

Anchoring against metrics and an evaluation set provides the following benefits:

1. You can iteratively and confidently refine your application's quality during development - no more vibe checks or guessing if a change resulted in an improvement.

2. Getting alignment with business stakeholders on the readiness of the application for production becomes more straightforward when you can confidently state, *"we know our application answers the most critical questions to our business correctly and doesn't hallucinate."*

*>> Evaluation-driven development is known in the academic research community as "hill climbing" akin to climbing a hill to reach the peak - where the hill is your metric and the peak is 100% accuracy on your evaluation set.*
