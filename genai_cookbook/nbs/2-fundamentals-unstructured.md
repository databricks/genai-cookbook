# Agent fundamentals

In [section 1](1-introduction-to-agents) of this guide, we introduced agents and RAG, explained its functionality at a high level, and highlighted its advantages over standalone LLMs.

This section will introduce the key components and principles behind developing agent applications over unstructured data. In particular, we will discuss:

1. **[Data pipeline](./2-fundamentals-unstructured-data-pipeline):** Transforming unstructured documents, such as collections of PDFs, into a format suitable for retrieval using the agent application's **data pipeline**.
2. [**Retrieval, Augmentation, and Generation (RAG chain)**](./2-fundamentals-unstructured-chain): A series (or **chain**) of steps is called to:
    1. Understand the user's question
    2. Retrieve the supporting data
    3. Call an LLM to generate a response based on the user's question and supporting data
3. [**Evaluation**](./2-fundamentals-unstructured-eval): Assessing the agent application to determine its quality/cost/latency to ensure it meets your business requirements.
4. [**Governance & LLMOps**](./2-fundamentals-unstructured-llmops): Tracking and managing the lifecycle of each component, including data lineage and governance (access controls).

```{image} ../images/2-fundamentals-unstructured/1_img.png
:alt: Major components of RAG over unstructured data
:align: center
```
<br/>