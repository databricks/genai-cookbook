## Root cause & iteratively fix quality issues

```{image} ../images/5-hands-on/14_img.png
:align: center
```

While a basic RAG chain is relatively straightforward to implement, refining it to consistently produce high-quality outputs is often non-trivial. Identifying the root causes of issues and determining which levers of the solution to pull to improve output quality requires understanding the various components and their interactions.

Simply vectorizing a set of documents, retrieving them via semantic search, and passing the retrieved documents to an LLM is not sufficient to guarantee optimal results. To yield high-quality outputs, you need to consider factors such as (but not limited to) chunking strategy of documents, choice of LLM and model parameters, or whether to include a query understanding step. As a result, ensuring high quality RAG outputs will generally involve iterating over both the data pipeline (e.g., chunking) and the RAG chain itself (e.g., choice of LLM).

This section is divided into 2 steps:

1. [Identify the root cause of quality issues](./5-hands-on-improve-quality-step-1.md)
2. [Implement and evaluate fixes to the identified root cause](./5-hands-on-improve-quality-step-2.md)
