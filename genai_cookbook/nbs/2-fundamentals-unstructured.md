# Section 2: Fundamentals of RAG over unstructured documents

In [section 1](1-introduction-to-rag) of this guide, we introduced RAG, explained its functionality at a high level, and highlighted its advantages over standalone LLMs.

This section will introduce the key components and principles behind developing RAG applications over unstructured data. In particular, we will discuss:

1. **[Data pipeline](#data-pipeline):** Transforming unstructured documents, such as collections of PDFs, into a format suitable for retrieval using the RAG application's **data pipeline**.
2. [**Retrieval, Augmentation, and Generation (RAG chain)**](#retrieval-augmentation-and-generation-rag-chain): A series (or **chain**) of steps is called to:
    1. Understand the user's question
    2. Retrieve the supporting data
    3. Call an LLM to generate a response based on the user's question and supporting data
3. [**Evaluation**](#evaluation-monitoring): Assessing the RAG application to determine its quality/cost/latency to ensure it meets your business requirements

```{image} ../images/2-fundamentals-unstructured/1_img.png
:alt: Major components of RAG over unstructured data
:align: center
```

## Data pipeline

Throughout this guide we will focus on preparing unstructured data for use in RAG applications. *Unstructured* data refers to data without a specific structure or organization, such as PDF documents that might include text and images, or multimedia content such as audio or videos.

Unstructured data lacks a predefined data model or schema, making it impossible to query on the basis of structure and metadata alone. As a result, unstructured data requires techniques that can understand and extract semantic meaning from raw text, images, audio, or other content.

During data preparation, the RAG application's data pipeline takes raw unstructured data and transforms it into discrete chunks that can be queried based on their relevance to a user's query. The key steps in data preprocessing are outlined below. Each step has a variety of knobs that can be tuned - for a deeper dive discussion on these knobs, please refer to the [deep dive into RAG section.](/nbs/3-deep-dive)

In the remainder of this section, we describe the process of preparing unstructured data for retrieval using *semantic search*. Semantic search understands the contextual meaning and intent of a user query to provide more relevant search results.

Semantic search is one of several approaches that can be taken when implementing the retrieval component of a RAG application over unstructured data. We cover alternate retrieval strategies in the [retrieval deep dive section](/nbs/3-deep-dive).

```{image} ../images/2-fundamentals-unstructured/2_img.png
:align: center
```

The following are the typical steps of a data pipeline in a RAG application using unstructured data:

1. **Parse the raw documents:** The initial step involves transforming raw data into a usable format. This can include extracting text, tables, and images from a collection of PDFs or employing optical character recognition (OCR) techniques to extract text from images.

2. **Extract document metadata (optional):** In some cases, extracting and using document metadata, such as document titles, page numbers, URLs, or other information can help the retrieval step more precisely query the correct data.

3. **Chunk documents:** To ensure the parsed documents can fit into the embedding model and the LLM's context window, we break the parsed documents into smaller, discrete chunks. Retrieving these focused chunks, rather than entire documents, gives the LLM more targeted content from which to generate its responses.

4. **Embedding chunks:** In a RAG application that uses semantic search, a special type of language model called an *embedding model* transforms each of the chunks from the previous step into numeric vectors, or lists of numbers, that encapsulate the meaning of each piece of content. Crucially, these vectors represent the semantic meaning of the text, not just surface-level keywords. This will later enable searching based on meaning rather than literal text matches.

5. **Index chunks in a vector database:** The final step is to load the vector representations of the chunks, along with the chunk's text, into a *vector database*. A vector database is a specialized type of database designed to efficiently store and search for vector data like embeddings. To maintain performance with a large number of chunks, vector databases commonly include a vector index that uses various algorithms to organize and map the vector embeddings in a way that optimizes search efficiency. At query time, a user's request is embedded into a vector, and the database leverages the vector index to find the most similar chunk vectors, returning the corresponding original text chunks.

The process of computing similarity can be computationally expensive. Vector indexes, such as [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html), speed this process up by providing a mechanism for efﬁciently organizing and navigating embeddings, often via sophisticated approximation methods. This enables rapid ranking of the most relevant results without comparing each embedding to the user's query individually.

Each step in the data pipeline involves engineering decisions that impact the RAG application's quality. For example, choosing the right chunk size in step (3) ensures the LLM receives specific yet contextualized information, while selecting an appropriate embedding model in step (4) determines the accuracy of the chunks returned during retrieval.

This data preparation process is referred to as *offline* data preparation, as it occurs before the system answers queries, unlike the *online* steps triggered when a user submits a query.

## Retrieval, augmentation, and generation (RAG Chain)

Once the data has been processed by the data pipeline, it is suitable for use in the RAG application. This section describes the process that occurs once the user submits a request to the RAG application in an online setting. The series, or *chain* of steps that are invoked at inference time is commonly referred to as the RAG chain.

```{image} ../images/2-fundamentals-unstructured/3_img.png
:align: center
```

1. **(Optional) User query preprocessing:** In some cases, the user's query is preprocessed to make it more suitable for querying the vector database. This can involve formatting the query within a template, using another model to rewrite the request, or extracting keywords to aid retrieval. The output of this step is a *retrieval query* which will be used in the subsequent retrieval step.

2. **Retrieval:** To retrieve supporting information from the vector database, the retrieval query is translated into an embedding using *the same embedding model* that was used to embed the document chunks during data preparation. These embeddings enable comparison of the semantic similarity between the retrieval query and the unstructured text chunks, using measures like cosine similarity. Next, chunks are retrieved from the vector database and ranked based on how similar they are to the embedded request. The top (most similar) results are returned.

3. **Prompt augmentation:** The prompt that will be sent to the LLM is formed by augmenting the user's query with the retrieved context, in a template that instructs the model how to use each component, often with additional instructions to control the response format. The process of iterating on the right prompt template to use is referred to as [prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering).

4. **LLM Generation**: The LLM takes the augmented prompt, which includes the user's query and retrieved supporting data, as input. It then generates a response that is grounded on the additional context.

5. **(Optional) Post-processing:** The LLM's response may be processed further to apply additional business logic, add citations, or otherwise refine the generated text based on predefined rules or constraints.

As with the RAG application data pipeline, there are numerous consequential engineering decisions that can affect the quality of the RAG chain. For example, determining how many chunks to retrieve in (2) and how to combine them with the user's query in (3) can both significantly impact the model's ability to generate quality responses.

Throughout the chain, various guardrails may be applied to ensure compliance with enterprise policies. This might involve filtering for appropriate requests, checking user permissions before accessing data sources, and applying content moderation techniques to the generated responses.

## Evaluation & monitoring

Evaluation and monitoring are critical components to understand if your RAG application is performing to the quality, cost, and latency requirements dictated by your use case. Evaluation happens during development and monitoring happens once the application is deployed to production.

RAG over unstructured data is a complex system with many components that impact the application's quality. Adjusting any single element can have cascading effects on the others. For instance, data formatting changes can influence the retrieved chunks and the LLM's ability to generate relevant responses. Therefore, it's crucial to evaluate each of the application's components in addition to the application as a whole in order to iteratively refine it based on those assessments.

Evaluation and monitoring the quality, cost and latency requires several components:

- **Defining quality with metrics**: You can't manage what you don't measure. In order to improve RAG quality, it is essential to define what quality means for your use case. Depending on the application, important metrics might include response accuracy, latency, cost, or ratings from key stakeholders.

- **Building an effective evaluation set:** To rigorously evaluate your RAG application, you need a curated set of evaluation queries (and ideally outputs) that are representative of the application's intended use. These evaluation examples should be challenging, diverse, and updated to reflect changing usage and requirements.

- **Monitoring application usage:** Instrumentation that tracks inputs, outputs, and intermediate steps such as document retrieval enables ongoing monitoring and early detection and diagnosis of issues that arise in development and production.

- **Collecting stakeholder feedback:** As a developer, you may not be a domain expert in the content of the application you are developing. In order to collect feedback from human experts who can assess your application quality, you need an interface that allows them to interact with the application and provide detailed feedback.

We will cover evaluation in much more detail in [Section 4: Evaluation](/nbs/4-evaluation).

The [next section](/nbs/3-deep-dive) of this guide will unpack the finer details of the typical components that make up the data pipeline and RAG chain of a RAG application using unstructured data.
