# Agents overview

This section provides an overview of Agents: what it is, how it works, and key concepts.

## What are AI agents and tools?

AI agents are systems where models make decisions, often using tools like Databricks' Unity Catalog functions toperform tasks such as retrieving data or interacting with external services.

See Databricks docs ([AWS](https://docs.databricks.com/en/generative-ai/ai-agents.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/ai-agents)) for more info.

## What is retrieval-augmented generation?

Retrieval-augmented generation (RAG) is a technique that enables a large language model (LLM) to generate enriched responses by augmenting a user’s prompt with supporting data retrieved from an outside information source. By incorporating this retrieved information, RAG enables the LLM to generate more accurate, higher quality responses compared to not augmenting the prompt with additional context.

For example, suppose you are building a question-and-answer chatbot to help employees answer questions about your company’s proprietary documents. A standalone LLM won’t be able to accurately answer questions about the content of these documents if it was not specifically trained on them. The LLM might refuse to answer due to a lack of information or, even worse, it might generate an incorrect response. 

RAG addresses this issue by first retrieving relevant information from the company documents based on a user’s query, and then providing the retrieved information to the LLM as additional context. This allows the LLM to generate a more accurate response by drawing from the specific details found in the relevant documents. In essence, RAG enables the LLM to “consult” the retrieved information to formulate its answer.

An agent with a retriever tool is one pattern for RAG, and has the advantage of deciding when to it needs to perform retrieval. This cookbook will describe how to build such an agent. 

## Core components of an agent application

An agent application is an example of a [compound AI system](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/): it expands on the language capabilities of the model alone by combining it with other tools and procedures.

When using a standalone LLM, a user submits a request, such as a question, to the LLM, and the LLM responds with an answer based solely on its training data.  

In its most basic form, the following steps happen in an agent application:

1. **User query understanding**: First the agent needs to use an LLM to understand the user's query. This step may also consider the previous steps of the conversation if provided.

2. **Tool selection**: The agent will use an LLM to determine if it should use a retriever tool. In the case of a vector search retriever, the LLM will create a retriever query, which will help retriever relevant chunks from the vector database. If no tool is selected, the agent will skip to step 4 and generate the final response.

3. **Tool execution**: The agent will then execute the tool with the parameters determined by the LLM and return the output. 

4. **LLM Generation**: The LLM will then generate the final response.

The image below demonstrates a RAG agent where a retrieval tool is selected.

```{image} ../images/1-introduction-to-agents/1_img.png
:alt: RAG process
:align: center
```

<br>

This is a simplified overview of the RAG process, but it's important to note that implementing an agent application involves a number of complex tasks. Preprocessing source data to make it suitable for retrieval, formatting the augmented prompt, and evaluating the generated responses all require careful consideration and effort. These topics will be covered in greater detail in later sections of this guide.

## Why use RAG?

The following table outlines the benefits of using RAG versus a stand-alone LLM:

| With an LLM alone                                                                                                                                                                                   | Using LLMs with RAG                                                                                                                                                                                                   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **No proprietary knowledge:** LLMs are generally trained on publicly available data, so they cannot accurately answer questions about a company's internal or proprietary data.                      | **RAG applications can incorporate proprietary data:** A RAG application can supply proprietary documents such as memos, emails, and design documents to an LLM, enabling it to answer questions about those documents. |
| **Knowledge isn't updated in real time:** LLMs do not have access to information about events that occurred after they were trained. For example, a standalone LLM cannot tell you anything about stock movements today. | **RAG applications can access real-time data:** A RAG application can supply the LLM with timely information from an updated data source, allowing it to provide useful answers about events past its training cutoff date.                                                                                                                                                                     |
| **Lack of citations:** LLMs cannot cite specific sources of information when responding, leaving the user unable to verify whether the response is factually correct or a [hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)). | **RAG can cite sources:** When used as part of a RAG application, an LLM can be asked to cite its sources.                                                                                                            |
| **Lack of data access controls (ACLs):** LLMs alone can't reliably provide different answers to different users based on specific user permissions.                                                   | **RAG allows for data security / ACLs:** The retrieval step can be designed to find only the information that the user has permission to access, enabling a RAG application to selectively retrieve personal or proprietary information based on the credentials of the individual user.                                                                                                                 |

## Types of RAG

The RAG architecture can work with 2 types of **supporting data**:

| | Structured data | Unstructured data |
|---|---|---|
| **Definition** | Tabular data arranged in rows & columns with a specific schema e.g., tables in a database. | Data without a specific structure or organization, e.g., documents that include text and images or multimedia content such as audio or videos. |
| **Example data sources** | - Customer records in a BI or Data Warehouse system<br>- Transaction data from a SQL database<br>- Data from application APIs (e.g., SAP, Salesforce, etc) | - PDFs<br>- Google/Office documents<br>- Wikis<br>- Images<br>- Videos |

Which data you use for your retriever depends on your use case. The remainder of this guide focuses on agents that use a retriever tool for unstructured data.
