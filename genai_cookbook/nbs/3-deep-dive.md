# Agent quality knobs

In the previous section, we introduced the key components of a RAG application and discussed the fundamental principles behind developing RAG applications over unstructured data. This section discusses how you can think about refining each component in order to increase the quality of your application.

There are myriad "knobs" to tune at every point in both the offline data pipeline, and online RAG chain. While there are countless others, we focus on the most important knobs that have the greatest impact on the the quality of your RAG application. Databricks recommends starting with these knobs.

From a conceptual point of view, it's helpful to view RAG quality knobs through the lens of the 2 key types of quality issues:

1. **Retrieval quality**
   - Are you retrieving the most relevant information for a given retrieval query?
      - It's difficult to generate high quality RAG output if the context provided to the LLM is missing important information or contains superfluous information.
2. **Generation quality**
   - Given the retrieved information and the original user query, is the LLM generating the most accurate, coherent, and helpful response possible?
      - Issues here can manifest as hallucinations, inconsistent output, or failure to directly address the user query.

RAG apps have two components that can be iterated on to address quality challenges: data pipeline and the chain.  It's tempting to assume a clean division between retrieval issues (simply update the data pipeline) and generation issues (update the RAG chain). However, the reality is more nuanced. Retrieval quality can be influenced by *both* the data pipeline (e.g., parsing/chunking strategy, metadata strategy, embedding model) and the RAG chain (e.g., user query transformation, number of chunks retrieved, re-ranking). Similarly, generation quality will invariably be impacted by poor retrieval (e.g., irrelevant or missing information affecting model output).

This overlap underscores the need for a holistic approach to RAG quality improvement. By understanding which components to change across both the data pipeline and RAG chain, and how these changes affect the overall solution, you can make targeted updates to improve RAG output quality.

[**Data pipeline**](/nbs/3-deep-dive-data-pipeline)

```{image} ../images/5-hands-on/15_img.png
:align: center
```
<br/>

- What is the composition of the input data corpus?
- How raw data is extracted and transformed into a usable format (e.g., parsing a PDF document)
- How documents are split into smaller chunks and how those chunks are formatted (e.g., chunking strategy, chunk size)
- What metadata (e.g., section title, document title) is extracted about each document/chunk? How is this metadata included (or not included) in each chunk?
- Which embedding model is used to convert text into vector representations for similarity search

[**RAG chain**](/nbs/3-deep-dive-chain)

```{image} ../images/5-hands-on/16_img.png
:align: center
```

<br/>

- The choice of LLM and its parameters (e.g., temperature, max tokens)
- The retrieval parameters (e.g., number of chunks/documents retrieved)
- The retrieval approach (e.g., keyword vs. hybrid vs. semantic search, rewriting the user's query, transforming a user's query into filters, re-ranking)
- How to format the prompt with retrieved context, to guide the LLM towards desired output
