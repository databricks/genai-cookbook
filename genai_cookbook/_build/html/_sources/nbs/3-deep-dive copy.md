# Deep dive into RAG over unstructured documents

In the previous section, we introduced the key components of a RAG application and discussed the fundamental principles behind developing RAG applications over unstructured data. This section discusses how you can think about refining each component in order to increase the quality of your application.

Above we alluded to the myriad of "knobs" to tune at every point in both the offline data pipeline, and online RAG chain. While there are numerous options to consider, we will focus on the table stakes considerations that should be prioritized when improving the quality of your RAG application. It's important to note that this is just scratching the surface, and there are many more advanced techniques that can be explored.

In the following sections of this guide, we will discuss how to measure changes with [evals](/nbs/4-evaluation), and finish by outlining how to diagnose root causes and possible fixes in the final [hands-on section](/nbs/5-hands-on).

## Data pipeline

```{image} ../images/3-deep-dive/1_img.png
:align: center
```

The foundation of any RAG application with unstructured data is the data pipeline. This pipeline is responsible for preparing the unstructured data in a format that can be effectively utilized by the RAG application. While this data pipeline can become arbitrarily complex, the following are the key components you need to think about when first building your RAG application:

1. **Corpus composition:** Selecting the right data sources and content based on the specific use case

2. **Parsing:** Extracting relevant information from the raw data using appropriate parsing techniques

3. **Chunking:** Breaking down the parsed data into smaller, manageable chunks for efficient retrieval

4. **Embedding:** Converting the chunked text data into a numerical vector representation that captures its semantic meaning

We discuss how to experiment with all of these data pipeline choices from a practical standpoint in [implementing data pipeline changes](/nbs/5-hands-on.md#data-pipeline-changes).

### Corpus composition

To state the obvious, without the right data corpus, your RAG application won't be able to retrieve the information required to answer a user query. The right data will be entirely dependent on the specific requirements and goals of your application, making it crucial to dedicate time to understand the nuances of data available (see the [requirements gathering section](/nbs/5-hands-on.md#requirements-questions) for guidance on this).

For example, when building a customer support bot, you might consider including:

- Knowledge base documents
- Frequently asked questions (FAQs)
- Product manuals and specifications
- Troubleshooting guides

Engage domain experts and stakeholders from the outset of any project to help identify and curate relevant content that could improve the quality and coverage of your data corpus. They can provide insights into the types of queries that users are likely to submit, and help prioritize the most important information to include.

### Parsing

Having identified the data sources for your RAG application, the next step will be extracting the required information from the raw data. This process, known as parsing, involves transforming the unstructured data into a format that can be effectively utilized by the RAG application.

The specific parsing techniques and tools you use will depend on the type of data you are working with. For example:

- **Text documents** (e.g., PDFs, Word docs): Off-the-shelf libraries like [unstructured](https://github.com/Unstructured-IO/unstructured) and [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/) can handle various file formats and provide options for customizing the parsing process.

- **HTML documents**: HTML parsing libraries like [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) can be used to extract relevant content from web pages. With these you can navigate the HTML structure, select specific elements, and extract the desired text or attributes.

- **Images and scanned documents**: Optical Character Recognition (OCR) techniques will typically be required to extract text from images. Popular OCR libraries include [Tesseract](https://github.com/tesseract-ocr/tesseract), [Amazon Textract](https://aws.amazon.com/textract/ocr/), [Azure AI Vision OCR](https://azure.microsoft.com/en-us/products/ai-services/ai-vision/), and [Google Cloud Vision API](https://cloud.google.com/vision).

When parsing your data, consider the following best practices:

1. **Data cleaning:** Preprocess the extracted text to remove any irrelevant or noisy information, such as headers, footers, or special characters. Be cognizant of reducing the amount of unnecessary or malformed information that your RAG chain will need to process.

2. **Handling errors and exceptions:** Implement error handling and logging mechanisms to identify and resolve any issues encountered during the parsing process. This will help you quickly identify and fix problems. Doing so often points to upstream issues with the quality of the source data.

3. **Customizing parsing logic:** Depending on the structure and format of your data, you may need to customize the parsing logic to extract the most relevant information. While it may require additional effort upfront, invest the time to do this if required - it often prevents a lot of downstream quality issues.

4. **Evaluating parsing quality**: Regularly assess the quality of the parsed data by manually reviewing a sample of the output. This can help you identify any issues or areas for improvement in the parsing process.

### Chunking

```{image} ../images/3-deep-dive/2_img.png
:align: center
```

After parsing the raw data into a more structured format, the next step is to break it down into smaller, manageable units called *chunks*. Segmenting large documents into smaller, semantically concentrated chunks, ensures that retrieved data fits in the LLM's context, while minimizing the inclusion of distracting or irrelevant information. The choices made on chunking will directly affect what retrieved data the LLM is provided, making it one of the first layers of optimization in a RAG application.

When chunking your data, you will generally need to consider the following factors:

1. **Chunking strategy:** The method you use to divide the original text into chunks. This can involve basic techniques such as splitting by sentences, paragraphs, or specific character/token counts, through to more advanced document-specific splitting strategies.

2. **Chunk size:** Smaller chunks may focus on specific details but lose some surrounding information. Larger chunks may capture more context but can also include irrelevant information.

3. **Overlap between chunks:** To ensure that important information is not lost when splitting the data into chunks, consider including some overlap between adjacent chunks. Overlapping can ensure continuity and context preservation across chunks.

4. **Semantic coherence:** When possible, aim to create chunks that are semantically coherent, meaning they contain related information and can stand on their own as a meaningful unit of text. This can be achieved by considering the structure of the original data, such as paragraphs, sections, or topic boundaries.

5. **Metadata:** Including relevant metadata within each chunk, such as the source document name, section heading, or product names can improve the retrieval process. This additional information in the chunk can help match retrieval queries to chunks.

Finding the right chunking method is both iterative and context-dependent. There is no one-size-fits all approach; the optimal chunk size and method will depend on the specific use case and the nature of the data being processed. Broadly speaking, chunking strategies can be viewed as the following:

- **Fixed-size chunking:** Split the text into chunks of a predetermined size, such as a fixed number of characters or tokens (e.g., [LangChain CharacterTextSplitter](https://python.langchain.com/v0.2/docs/how_to/character_text_splitter/)). While splitting by an arbitrary number of characters/tokens is quick and easy to set up, it will typically not result in consistent semantically coherent chunks.

- **Paragraph-based chunking:** Use the natural paragraph boundaries in the text to define chunks. This method can help preserve the semantic coherence of the chunks, as paragraphs often contain related information (e.g, [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/)).

- **Format-specific chunking:** Formats such as markdown or HTML have an inherent structure within them which can be used to define chunk boundaries (for example, markdown headers). Tools like LangChain's [MarkdownHeaderTextSplitter](https://python.langchain.com/v0.2/docs/how_to/markdown_header_metadata_splitter/#how-to-return-markdown-lines-as-separate-documents) or HTML [header](https://python.langchain.com/v0.2/docs/how_to/HTML_header_metadata_splitter/)/[section](https://python.langchain.com/v0.2/docs/how_to/HTML_section_aware_splitter/)-based splitters can be used for this purpose.

- **Semantic chunking:** Techniques such as topic modeling can be applied to identify semantically coherent sections within the text. These approaches analyze the content or structure of each document to determine the most appropriate chunk boundaries based on shifts in topic. Although more involved than more basic approaches, semantic chunking can help create chunks that are more aligned with the natural semantic divisions in the text (see [LangChain SemanticChunker](https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/) for an example of this).

```{image} ../images/3-deep-dive/3_img.png
:align: center
```

**Example:** Fixed-size chunking example using LangChain's [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) with `chunk_size=100` and `chunk_overlap=20`. [ChunkViz](https://chunkviz.up.railway.app/) provides an interactive way to visualize how different chunk size and chunk overlap values with Langchain's character splitters affects resulting chunks.

### Embedding model

```{image} ../images/3-deep-dive/4_img.png
:align: center
```

After chunking your data, the next step is to convert the text chunks into a vector representation using an embedding model. An embedding model is used to convert each text chunk into a vector representation that captures its semantic meaning. By representing chunks as dense vectors, embeddings allow for fast and accurate retrieval of the most relevant chunks based on their semantic similarity to a retrieval query. At query time, the retrieval query will be transformed using the same embedding model that was used to embed chunks in the data pipeline.

When selecting an embedding model, consider the following factors:

- **Model choice:** Each embedding model has its nuances, and the available benchmarks may not capture the specific characteristics of your data. Experiment with different off-the-shelf embedding models, even those that may be lower-ranked on standard leaderboards like [MTEB](https://huggingface.co/spaces/mteb/leaderboard). Some examples to consider include:
  - [GTE-Large-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
  - [OpenAI's text-embedding-ada-002, text-embedding-large, and text-embedding-small](https://platform.openai.com/docs/guides/embeddings)

- **Max tokens:** Be aware of the maximum token limit for your chosen embedding model. If you pass chunks that exceed this limit, they will be truncated, potentially losing important information. For example, [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) has a maximum token limit of 512.

- **Model size:** Larger embedding models generally offer better performance but require more computational resources. Strike a balance between performance and efficiency based on your specific use case and available resources.

- **Fine-tuning:** If your RAG application deals with domain-specific language (e.g., internal company acronyms or terminology), consider fine-tuning the embedding model on domain-specific data. This can help the model better capture the nuances and terminology of your particular domain, and can often lead to improved retrieval performance.

## RAG Chain

```{image} ../images/3-deep-dive/5_img.png
:align: center
```

The RAG chain takes a user query as input, retrieves relevant information given that query, and generates an appropriate response grounded on the retrieved data. While the exact steps within a RAG chain can vary widely depending on the use case and requirements, the following are the key components to consider when building your RAG chain:

1. **Query understanding:** Analyzing and transforming user queries to better represent intent and extract relevant information, such as filters or keywords, to improve the retrieval process.

2. **Retrieval:** Finding the most relevant chunks of information given a retrieval query. In the unstructured data case, this typically involves one or a combination of semantic or keyword-based search.

3. **Prompt augmentation:** Combining a user query with retrieved information and instructions to guide the LLM towards generating high-quality responses.

4. **LLM:** Selecting the most appropriate model (and model parameters) for your application to optimize/balance performance, latency, and cost.

5. **Post-processing and guardrails:** Applying additional processing steps and safety measures to ensure the LLM-generated responses are on-topic, factually consistent, and adhere to specific guidelines or constraints.

In the [implementing RAG chain changes](nbs/5-hands-on.md#rag-chain-changes) section we will demonstrate how to iterate over these various components of a chain.

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

   Filter extraction must be done in conjunction with changes to both metadata extraction [data pipeline] and retrieval [RAG chain] components. The metadata extraction step should ensure that the relevant metadata fields are available for each document/chunk, and the retrieval step should be implemented to accept and apply extracted filters.

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
