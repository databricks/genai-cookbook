## Retrieval, augmentation, and generation (aka RAG Agent)

Once the data has been processed by the data pipeline, it is suitable for use in a retriever tool. This section describes the process that occurs once the user submits a request to the agent application in an online setting.

<!-- TODO (prithvi): add this back in once updated to agents
```{image} ../images/2-fundamentals-unstructured/3_img.png
:align: center
``` -->
<br/>

1. **User query understanding**: First the agent needs to use an LLM to understand the user's query. This step may also consider the previous steps of the conversation if provided.

2. **Tool selection**: The agent will use an LLM to determine if it should use a retriever tool. In the case of a vector search retriever, the LLM will create a retriever query, which will help retriever relevant chunks from the vector database. If no tool is selected, the agent will skip to step 4 and generate the final response.

3. **Tool execution**: The agent will then execute the tool with the parameters determined by the LLM and return the output. 

4. **LLM Generation**: The LLM will then generate the final response.

As with the retriever data pipeline, there are numerous consequential engineering decisions that can affect the quality of the agent. For example, determining how many chunks to retrieve in and when to select the retriever tool can both significantly impact the model's ability to generate quality responses.

Throughout the agent, various guardrails may be applied to ensure compliance with enterprise policies. This might involve filtering for appropriate requests, checking user permissions before accessing data sources, and applying content moderation techniques to the generated responses.
