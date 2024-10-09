## Data pipeline

Throughout this guide we will focus on preparing unstructured data for use in Agent applications. *Unstructured* data refers to data without a specific structure or organization, such as PDF documents that might include text and images, or multimedia content such as audio or videos.

Unstructured data lacks a predefined data model or schema, making it impossible to query on the basis of structure and metadata alone. As a result, unstructured data requires techniques that can understand and extract semantic meaning from raw text, images, audio, or other content.

During data preparation, the Agent application's data pipeline takes raw unstructured data and transforms it into discrete chunks that can be queried based on their relevance to a user's query. The key steps in data preprocessing are outlined below. Each step has a variety of knobs that can be tuned - for a deeper dive discussion on these knobs, please refer to the [deep dive into RAG section.](/nbs/3-deep-dive)

```{image} ../images/2-fundamentals-unstructured/2_img.png
:align: center
```
<br/>

In the remainder of this section, we describe the process of preparing unstructured data for retrieval using *semantic search*. Semantic search understands the contextual meaning and intent of a user query to provide more relevant search results.

Semantic search is one of several approaches that can be taken when implementing the retrieval component of a RAG application over unstructured data. We cover alternate retrieval strategies in the [retrieval knobs section](/nbs/3-deep-dive-chain.md#retrieval).



The following are the typical steps of a data pipeline in an agent application using unstructured data:

1. **Parse the raw documents:** The initial step involves transforming raw data into a usable format. This can include extracting text, tables, and images from a collection of PDFs or employing optical character recognition (OCR) techniques to extract text from images.

2. **Extract document metadata (optional):** In some cases, extracting and using document metadata, such as document titles, page numbers, URLs, or other information can help the retrieval step more precisely query the correct data.

3. **Chunk documents:** To ensure the parsed documents can fit into the embedding model and the LLM's context window, we break the parsed documents into smaller, discrete chunks. Retrieving these focused chunks, rather than entire documents, gives the LLM more targeted context from which to generate its responses.

4. **Embedding chunks:** In a RAG application that uses semantic search, a special type of language model called an *embedding model* transforms each of the chunks from the previous step into numeric vectors, or lists of numbers, that encapsulate the meaning of each piece of content. Crucially, these vectors represent the semantic meaning of the text, not just surface-level keywords. This will later enable searching based on meaning rather than literal text matches.

5. **Index chunks in a vector database:** The final step is to load the vector representations of the chunks, along with the chunk's text, into a *vector database*. A vector database is a specialized type of database designed to efficiently store and search for vector data like embeddings. To maintain performance with a large number of chunks, vector databases commonly include a vector index that uses various algorithms to organize and map the vector embeddings in a way that optimizes search efficiency. At query time, a user's request is embedded into a vector, and the database leverages the vector index to find the most similar chunk vectors, returning the corresponding original text chunks.

The process of computing similarity can be computationally expensive. Vector indexes, such as [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html), speed this process up by providing a mechanism for efﬁciently organizing and navigating embeddings, often via sophisticated approximation methods. This enables rapid ranking of the most relevant results without comparing each embedding to the user's query individually.

Each step in the data pipeline involves engineering decisions that impact the agent application's quality. For example, choosing the right chunk size in step (3) ensures the LLM receives specific yet contextualized information, while selecting an appropriate embedding model in step (4) determines the accuracy of the chunks returned during retrieval.

This data preparation process is referred to as *offline* data preparation, as it occurs before the system answers queries, unlike the *online* steps triggered when a user submits a query.
