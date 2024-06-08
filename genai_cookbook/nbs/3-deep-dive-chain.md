## Retrieval, augmentation, and generation (aka RAG Chain)

```{image} ../images/3-deep-dive/5_img.png
:align: center
```

The RAG chain takes a user query as input, retrieves relevant information given that query, and generates an appropriate response grounded on the retrieved data. While the exact steps within a RAG chain can vary widely depending on the use case and requirements, the following are the key components to consider when building your RAG chain:

1. **Query understanding:** Analyzing and transforming user queries to better represent intent and extract relevant information, such as filters or keywords, to improve the retrieval process.

2. **Retrieval:** Finding the most relevant chunks of information given a retrieval query. In the unstructured data case, this typically involves one or a combination of semantic or keyword-based search.

3. **Prompt augmentation:** Combining a user query with retrieved information and instructions to guide the LLM towards generating high-quality responses.

4. **LLM:** Selecting the most appropriate model (and model parameters) for your application to optimize/balance performance, latency, and cost.

5. **Post-processing and guardrails:** Applying additional processing steps and safety measures to ensure the LLM-generated responses are on-topic, factually consistent, and adhere to specific guidelines or constraints.

In the [implementing RAG chain changes](/nbs/5-hands-on-improve-quality-step-2.md#rag-chain-changes) section we will demonstrate how to iterate over these various components of a chain.

### Query understanding

Using the user query directly as a retrieval query can work for some queries. However, it is generally beneficial to reformulate the query before the retrieval step. Query understanding comprises a step (or series of steps) at the beginning of a chain to analyze and transform user queries to better represent intent, extract relevant information, and ultimately help the subsequent retrieval process. Approaches to transforming a user query to improve retrieval include:

1. **Query Rewriting:** Query rewriting involves translating a user query into one or more queries that better represent the original intent. The goal is to reformulate the query in a way that increases the likelihood of the retrieval step finding the most relevant documents. This can be particularly useful when dealing with complex or ambiguous queries that might not directly match the terminology used in the retrieval documents.
<br>
   **Examples**:

   - Paraphrasing conversation history in a multi-turn chat
   - Correcting spelling mistakes in the user's query
   - Replacing words or phrases in the user query with synonyms to capture a broader range of relevant documents

```{eval-rst}
.. note::

   Query rewriting must be done in conjunction with changes to the retrieval component

.. include:: ./include-rst.rst
```

2. **Filter extraction:** In some cases, user queries may contain specific filters or criteria that can be used to narrow down the search results. Filter extraction involves identifying and extracting these filters from the query and passing them to the retrieval step as additional parameters. This can help improve the relevance of the retrieved documents by focusing on specific subsets of the available data.
<br>
   **Examples**:

   - Extracting specific time periods mentioned in the query, such as "articles from the last 6 months" or "reports from 2023".
   - Identifying mentions of specific products, services, or categories in the query, such as "Databricks Professional Services" or "laptops".
   - Extracting geographic entities from the query, such as city names or country codes.


```{eval-rst}
.. note::

   Filter extraction must be done in conjunction with changes to both metadata extraction [data pipeline](./3-deep-dive-data-pipeline.md) and [retriever chain](#retrieval) components. The metadata extraction step should ensure that the relevant metadata fields are available for each document/chunk, and the retrieval step should be implemented to accept and apply extracted filters.

.. include:: ./include-rst.rst
```

In addition to query rewriting and filter extraction, another important consideration in query understanding is whether to use a single LLM call or multiple calls. While using a single call with a carefully crafted prompt can be efficient, there are cases where breaking down the query understanding process into multiple LLM calls can lead to better results. This, by the way, is a generally applicable rule of thumb when you are trying to implement a number of complex logic steps into a single prompt.

For example, you might use one LLM call to classify the query intent, another to extract relevant entities, and a third to rewrite the query based on the extracted information. Although this approach may add some latency to the overall process, it can allow for more fine-grained control and potentially improve the quality of the retrieved documents.

Here's how a multi-step query understanding component might look for our a customer support bot:

1. **Intent classification:** Use an LLM to classify the user's query into predefined categories, such as "product information", "troubleshooting", or "account management".

2. **Entity extraction:** Based on the identified intent, use another LLM call to extract relevant entities from the query, such as product names, reported errors, or account numbers.

3. **Query rewriting:** Use the extracted intent and entities to rewrite the original query into a more specific and targeted format, e.g., "My RAG chain is failing to deploy on Model Serving, I'm seeing the following error...".

### Retrieval

The retrieval component of the RAG chain is responsible for finding the most relevant chunks of information given a retrieval query. In the context of unstructured data, retrieval typically involves one or a combination of semantic search, keyword-based search, and metadata filtering. The choice of retrieval strategy depends on the specific requirements of your application, the nature of the data, and the types of queries you expect to handle. Let's compare these options:

1. **Semantic search:** Semantic search uses an embedding model to convert each chunk of text into a vector representation that captures its semantic meaning. By comparing the vector representation of the retrieval query with the vector representations of the chunks, semantic search can retrieve documents that are conceptually similar, even if they don't contain the exact keywords from the query.

2. **Keyword-based search:** Keyword-based search determines the relevance of documents by analyzing the frequency and distribution of shared words between the retrieval query and the indexed documents. The more often the same words appear in both the query and a document, the higher the relevance score assigned to that document.

3. **Hybrid search:** Hybrid search combines the strengths of both semantic and keyword-based search by employing a two-step retrieval process. First, it performs a semantic search to retrieve a set of conceptually relevant documents. Then, it applies keyword-based search on this reduced set to further refine the results based on exact keyword matches. Finally, it combines the scores from both steps to rank the documents.

The following table contrasts each of these retrieval strategies against one another:

| | Semantic search | Keyword search | Hybrid search |
|---|---|---|---|
| **Simple explanation** | If the same **concepts** appear in the query and a potential document, they are relevant. | If the same **words** appear in the query and a potential document, they are relevant. The **more words** from the query in the document, the more relevant that document is. | Runs BOTH a semantic search and keyword search, then combines the results. |
| **Example use case** | Customer support where user queries are different than the words in the product manuals<br><br>e.g., *"how do i turn my phone on?"* and the manual section is called *"toggling the power"*. | Customer support where queries contain specific, non descriptive technical terms.<br><br>e.g., *"what does model HD7-8D do?"* | Customer support queries that combined both semantic and technical terms.<br><br>e.g., *"how do I turn on my HD7-8D?"* |
| **Technical approaches** | Uses embeddings to represent text in a continuous vector space, enabling semantic search | Relies on discrete token-based methods like [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model), [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) for keyword matching. | Use a re-ranking approach to combine the results, such as [reciprocal rank fusion](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) or a [re-ranking model](https://en.wikipedia.org/wiki/Learning_to_rank). |
| **Strengths** | Retrieving contextually similar information to a query, even if the exact words are not used. | Scenarios requiring precise keyword matches, ideal for specific term-focused queries such as product names. | Combines the best of both approaches. |

In addition to these core retrieval strategies, there are several techniques you can apply to further enhance the retrieval process:

- **Query expansion:** Query expansion can help capture a broader range of relevant documents by using multiple variations of the retrieval query. This can be achieved by either conducting individual searches for each expanded query, or using a concatenation of all expanded search queries in a single retrieval query.

> ***Note:** Query expansion must be done in conjunction with changes to the query understanding component [RAG chain]. The multiple variations of a retrieval query are typically generated in this step.*

- **Re-ranking:** After retrieving an initial set of chunks, apply additional ranking criteria (e.g., sort by time) or a reranker model to re-order the results. Re-ranking can help prioritize the most relevant chunks given a specific retrieval query. Reranking with cross-encoder models such as [mxbai-rerank](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) and [ColBERTv2](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/ColbertRerank/) can yield an uplift in retrieval performance.

- **Metadata filtering:** Use metadata filters extracted from the query understanding step to narrow down the search space based on specific criteria. Metadata filters can include attributes like document type, creation date, author, or domain-specific tags. By combining metadata filters with semantic or keyword-based search, you can create more targeted and efficient retrieval.

> ***Note:** Metadata filtering must be done in conjunction with changes to the query understanding [RAG chain] and metadata extraction [data pipeline] components.*

### Prompt augmentation

Prompt augmentation is the step where the user query is combined with the retrieved information and instructions in a prompt template to guide the language model towards generating high-quality responses. Iterating on this template to optimize the prompt provided to the LLM (i.e., prompt engineering) will be required to ensure that the model is guided to produce accurate, grounded, and coherent responses.

There are entire [guides to prompt engineering](https://www.promptingguide.ai/), but here are a number of considerations to keep in mind when you're iterating on the prompt template:

1. Provide examples
   - Include examples of well-formed queries and their corresponding ideal responses within the prompt template itself (i.e., [few-shot learning](https://arxiv.org/abs/2005.14165)). This helps the model understand the desired format, style, and content of the responses.
   - One useful way to come up with good examples is to identify types of queries your chain struggles with. Create gold-standard responses for those queries and include them as examples in the prompt.
   - Ensure that the examples you provide are representative of user queries you anticipate at inference time. Aim to cover a diverse range of expected queries to help the model generalize better.

2. Parameterize your prompt template
   - Design your prompt template to be flexible by parameterizing it to incorporate additional information beyond the retrieved data and user query. This could be variables such as current date, user context, or other relevant metadata.
   - Injecting these variables into the prompt at inference time can enable more personalized or context-aware responses.

3. Consider Chain-of-Thought prompting
   - For complex queries where direct answers aren't readily apparent, consider [Chain-of-Thought (CoT) prompting](https://arxiv.org/abs/2201.11903). This prompt engineering strategy breaks down complicated questions into simpler, sequential steps, guiding the LLM through a logical reasoning process.
   - By prompting the model to "think through the problem step-by-step," you encourage it to provide more detailed and well-reasoned responses, which can be particularly effective for handling multi-step or open-ended queries.

4. Prompts may not transfer across models
   - Recognize that prompts often do not transfer seamlessly across different language models. Each model has its own unique characteristics where a prompt that works well for one model may not be as effective for another.
   - Experiment with different prompt formats and lengths, refer to online guides (e.g., [OpenAI Cookbook](https://cookbook.openai.com/), [Anthropic cookbook](https://github.com/anthropics/anthropic-cookbook)), and be prepared to adapt and refine your prompts when switching between models.

### LLM

The generation component of the RAG chain takes the augmented prompt template from the previous step and passes it to a LLM. When selecting and optimizing an LLM for the generation component of a RAG chain, consider the following factors, which are equally applicable to any other steps that involve LLM calls:

1. Experiment with different off-the-shelf models
   - Each model has its own unique properties, strengths, and weaknesses. Some models may have a better understanding of certain domains or perform better on specific tasks.
   - As mentioned prior, keep in mind that the choice of model may also influence the prompt engineering process, as different models may respond differently to the same prompts.
   - If there are multiple steps in your chain that require an LLM, such as calls for query understanding in addition to the generation step, consider using different models for different steps. More expensive, general-purpose models may be overkill for tasks like determining the intent of a user query.

2. Start small and scale up as needed
   - While it may be tempting to immediately reach for the most powerful and capable models available (e.g., GPT-4, Claude), it's often more efficient to start with smaller, more lightweight models.
   - In many cases, smaller open-source alternatives like Llama 3 or DBRX can provide satisfactory results at a lower cost and with faster inference times. These models can be particularly effective for tasks that don't require highly complex reasoning or extensive world knowledge.
   - As you develop and refine your RAG chain, continuously assess the performance and limitations of your chosen model. If you find that the model struggles with certain types of queries or fails to provide sufficiently detailed or accurate responses, consider scaling up to a more capable model.
   - Monitor the impact of changing models on key metrics such as response quality, latency, and cost to ensure that you're striking the right balance for the requirements of your specific use case.

3. Optimize model parameters
   - Experiment with different parameter settings to find the optimal balance between response quality, diversity, and coherence. For example, adjusting the temperature can control the randomness of the generated text, while max_tokens can limit the response length.
   - Be aware that the optimal parameter settings may vary depending on the specific task, prompt, and desired output style. Iteratively test and refine these settings based on evaluation of the generated responses.

4. Task-specific fine-tuning
   - As you refine performance, consider fine-tuning smaller models for specific sub-tasks within your RAG chain, such as query understanding.
   - By training specialized models for individual tasks with the RAG chain, you can potentially improve the overall performance, reduce latency, and lower inference costs compared to using a single large model for all tasks.

5. Continued pre-training
   - If your RAG application deals with a specialized domain or requires knowledge that is not well-represented in the pre-trained LLM, consider performing continued pre-training (CPT) on domain-specific data.
   - Continued pre-training can improve a model's understanding of specific terminology or concepts unique to your domain. In turn this can reduce the need for extensive prompt engineering or few-shot examples.

### Post-processing & guardrails

After the LLM generates a response, it is often necessary to apply post-processing techniques or guardrails to ensure that the output meets the desired format, style, and content requirements. This final step (or multiple steps) in the chain can help maintain consistency and quality across the generated responses. If you are implementing post-processing and guardrails, consider some of the following:

1. Enforcing output format
   - Depending on your use case, you may require the generated responses to adhere to a specific format, such as a structured template or a particular file type (e.g., JSON, HTML, Markdown etc).
   - If structured output is required, libraries such as [Instructor](https://github.com/jxnl/instructor) or [Outlines](https://github.com/outlines-dev/outlines) provide good starting points to implement this kind of validation step.
   - When developing, take time to ensure that the post-processing step is flexible enough to handle variations in the generated responses while maintaining the required format.

2. Maintaining style consistency
   - If your RAG application has specific style guidelines or tone requirements (e.g., formal vs. casual, concise vs. detailed), a post-processing step can both check and enforce these style attributes across generated responses.

3. Content filters and safety guardrails
   - Depending on the nature of your RAG application and the potential risks associated with generated content, it may be important to [implement content filters or safety guardrails](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) to prevent the output of inappropriate, offensive, or harmful information.
   - Consider using models like [Llama Guard](https://marketplace.databricks.com/details/a4bc6c21-0888-40e1-805e-f4c99dca41e4/Databricks_Llama-Guard-Model) or APIs specifically designed for content moderation and safety, such as [OpenAI's moderation API](https://platform.openai.com/docs/guides/moderation), to implement safety guardrails.

4. Handling hallucinations
   - Defending against hallucinations can also be implemented as a post-processing step. This may involve cross-referencing the generated output with retrieved documents, or using additional LLMs to validate the factual accuracy of the response.
   - Develop fallback mechanisms to handle cases where the generated response fails to meet the factual accuracy requirements, such as generating alternative responses or providing disclaimers to the user.

5. Error handling
   - With any post-processing steps, implement mechanisms to gracefully deal with cases where the step encounters an issue or fails to generate a satisfactory response. This could involve generating a default response, or escalating the issue to a human operator for manual review.
