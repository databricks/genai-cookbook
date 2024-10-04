## Retrieval, augmentation, and generation (aka RAG Agent)

Once the data has been processed by the data pipeline, it is suitable for use in the retreival tool. This section describes the process that occurs once the user submits a request to the Agent application in an online setting. The series, or *chain* of steps that are invoked at inference time is commonly referred to as the Agent loop.

```{image} ../images/2-fundamentals-unstructured/3_img.png
:align: center
```
<br/>

1. **(Optional) User query preprocessing:** In some cases, the user's query is preprocessed to make it more suitable for querying the vector database. This can involve formatting the query within a template, using another model to rewrite the request, or extracting keywords to aid retrieval. The output of this step is a *retrieval query* which will be used in the subsequent retrieval step.

2. **Retrieval:** To retrieve supporting information from the vector database, the retrieval query is translated into an embedding using *the same embedding model* that was used to embed the document chunks during data preparation. These embeddings enable comparison of the semantic similarity between the retrieval query and the unstructured text chunks, using measures like cosine similarity. Next, chunks are retrieved from the vector database and ranked based on how similar they are to the embedded request. The top (most similar) results are returned.

3. **Prompt augmentation:** The prompt that will be sent to the LLM is formed by augmenting the user's query with the retrieved context, in a template that instructs the model how to use each component, often with additional instructions to control the response format. The process of iterating on the right prompt template to use is referred to as [prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering).

4. **LLM Generation**: The LLM takes the augmented prompt, which includes the user's query and retrieved supporting data, as input. It then generates a response that is grounded on the additional context.

5. **(Optional) Post-processing:** The LLM's response may be processed further to apply additional business logic, add citations, or otherwise refine the generated text based on predefined rules or constraints.

As with the retriever tool data pipeline, there are numerous consequential engineering decisions that can affect the quality of the Agent. For example, determining how many chunks to retrieve in (2) and how to combine them with the user's query in (3) can both significantly impact the model's ability to generate quality responses.

Throughout the chain, various guardrails may be applied to ensure compliance with enterprise policies. This might involve filtering for appropriate requests, checking user permissions before accessing data sources, and applying content moderation techniques to the generated responses.
