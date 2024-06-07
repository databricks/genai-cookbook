---
title: Databricks Generative AI Cookbook
---

# Databricks Mosaic Generative AI Cookbook

## RAG Guide
Follow this guide to learn how RAG works, the required components of a production-ready, high-quality RAG application, and Databricks’ recommended developer workflow for delivering a high-quality RAG application that uses unstructured data.  

This guide assumes you have selected a use case that requires unstructured docs RAG.

This guide is broken into 5 main sections:

1. Introduction to retrieval-augmented generation (RAG)
   - Overview of the RAG technique
   - Key benefits of using RAG
   - Types of RAG
2. Fundamentals of RAG over Unstructured Documents
   - Understanding retrieval
   - Components of a RAG application
3. Deep dive into RAG over Unstructured Documents
   - Deep dive into the components of a RAG application
4. How to evaluate RAG applications
   - Metrics to measure quality / cost / latency
   - Developing an evaluation set to measure quality
   - Infrastructure required to evaluate RAG apps
5. Practical hands-on guide to implementing high-quality RAG
   - Databricks recommended developer workflow for building a RAG application
   - How to iteratively improve the application's quality
   - Throughout, this section is supported by ready-to-use code examples

If you are less familiar with RAG, we suggest starting with section 1. If you are already familiar with RAG or simply want to get started quickly, start with section 5.  Section 5 refers back to the previous sections as needed to explain concepts.

## Featured Notebooks

::::{grid} 3
:class-container: text-center

:::{grid-item-card}
:link: /nbs/1-introduction-to-rag
:link-type: doc
:class-header: bg-light

Section 1: Introduction to retrieval-augmented generation (RAG)
^^^
Learn the basic concepts of retrieval-augmented generation (RAG) in this introdoctory notebook.
:::

:::{grid-item-card}
:link: /nbs/2-fundamentals-unstructured
:link-type: doc
:class-header: bg-light

Section 2: Fundamentals of RAG over Unstructured Documents
^^^
Introduction to the key components and principles of developing RAG applications over unstructured data.
:::

:::{grid-item-card}
:link: /nbs/3-deep-dive
:link-type: doc
:class-header: bg-light

Section 3: Deep dive into RAG over unstructured documents
^^^
A more detailed guide to refining each component of a RAG application over unstructured data.
:::
::::

```{tableofcontents}
```



