### Identify the root cause of quality issues

#### Retrieval quality

##### Debugging retrieval quality

Retrieval quality is arguably the most important component of a RAG application. If the most relevant chunks are not returned for a given query, the LLM will not have access to the necessary information to generate a high-quality response. Poor retrieval can thus lead to irrelevant, incomplete, or hallucinated output.

As discussed in [Section 4: Evaluation](/nbs/4-evaluation), metrics such as precision and recall can be calculated using a set of evaluation queries and corresponding ground-truth chunks/documents. If evaluation results indicate that relevant chunks are not being returned, you will need to investigate further to identify the root cause. This step requires manual effort to analyze the underlying data. With Mosaic AI, this becomes considerably easier given the tight integration between the data platform (Unity Catalog and Vector Search), and experiment tracking (MLflow LLM evaluation and MLflow tracing).

Here's a step-by-step process to address **retrieval quality** issues:

1. Identify a set of test queries with low retrieval quality metrics.

2. For each query, manually examine the retrieved chunks and compare them to the ground-truth retrieval documents.

3. Look for patterns or common issues among the queries with low retrieval quality. Some examples might include:
   - Relevant information is missing from the vector database entirely
   - Insufficient number of chunks/documents returned for a retrieval query
   - Chunks are too small and lack sufficient context
   - Chunks are too large and contain multiple, unrelated topics
   - The embedding model fails to capture semantic similarity for domain-specific terms

4. Based on the identified issue, hypothesize potential root causes and corresponding fixes. See the "[Common reasons for poor retrieval quality](#common-reasons-for-poor-retrieval-quality)" table below for guidance on this.

5. Implement the proposed fix for the most promising or impactful root cause, following [step 3](#step-3-implement-and-evaluate-changes). This may involve modifying the data pipeline (e.g., adjusting chunk size, trying a different embedding model) or the RAG chain (e.g., implementing hybrid search, retrieving more chunks).

6. Re-run the evaluation on the updated system and compare the retrieval quality metrics to the previous version. Once retrieval quality is at a desired level, proceed to evaluating generation quality (see [Debugging generation quality](#debugging-generation-quality)).

7. If retrieval quality is still not satisfactory, repeat steps 4-6 for the next most promising fixes until the desired performance is achieved.

##### Common reasons for poor retrieval quality

Each of these potential fixes are tagged as one of three types. Based on the type of change, you will follow different steps in section 3.

<!-- ![data-pipeline](../images/5-hands-on/data_pipeline.png) -->


<table>
<thead>
<tr>
<th>Retrieval Issue</th>
<th>Debugging Steps</th>
<th>Potential Fix</th>
</tr>
</thead>
<tbody>
<tr>
<td>Chunks are too small</td>
<td><ul><li>Examine chunks for incomplete cut-off information</li></ul></td>
<td><ul><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/>  Increase chunk size and/or overlap</li><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Try different chunking strategy</li></ul></td>
</tr>
<tr>
<td>Chunks are too large</td>
<td><ul><li>Check if retrieved chunks contain multiple, unrelated topics</td>
<td><ul><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Decrease chunk size</li><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Improve chunking strategy to avoid mixture of unrelated topics (e.g., semantic chunking)</li></ul></td>
</tr>
<tr>
<td>Chunks don&#39;t have enough information about the text from which they were taken</td>
<td><ul><li>Assess if the lack of context for each chunk is causing confusion or ambiguity in the retrieved results</li></ul></td>
<td><ul><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Chain codeTry adding metadata &amp; titles to each chunk (e.g., section titles)</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Retrieve more chunks, and use an LLM with larger context size</li></ul></td>
</tr>
<tr>
<td>Embedding model doesn&#39;t accurately understand the domain and/or key phrases in user queries</td>
<td><ul><li>Check if semantically similar chunks are being retrieved for the same query</li></ul></td>
<td><ul><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Try different embedding models</li><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Fine-tune embedding model on domain-specific data</li></ul></td>
</tr>
<tr>
<td>Limited retrieval quality due to embedding model&#39;s lack of domain understanding</td>
<td><ul><li>Look at retrieved results to check if they are semantically relevant but miss key domain-specific information</li></ul></td>
<td><ul><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Hybrid search</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Over-fetch retrieval results, and re-rank. Only feed top re-ranked results into the LLM context</li></ul></td>
</tr>
<tr>
<td>Relevant information missing from the vector database</td>
<td><ul><li>Check if any relevant documents or sections are missing from the vector database</li></ul></td>
<td><ul><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Add more relevant documents to the vector database</li><li><img src="../_images/data_pipeline.png" alt="data-pipeline" height="20"/> Improve document parsing and metadata extraction</li></ul></td>
</tr>
<tr>
<td>Retrieval queries are poorly formulated</td>
<td><ul><li>If user queries are being directly used for semantic search, analyze these queries and check for ambiguity, or lack of specificity. This can happen easily in multi-turn conversations where the raw user query references previous parts of the conversation, making it unsuitable to use directly as a retrieval query.</li><li>Check if query terms match terminology used in the search corpus</li></ul></td>
<td><ul><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Add query expansion or transformation approaches (i.e., given a user query, transform the query prior to semantic search)</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Add query understanding to identify intent and entities (e.g., use an LLM to extract properties to use in metadata filtering)</li></ul></td>
</tr>
</tbody>
</table>


#### Generation quality

##### Debugging generation quality

Even with optimal retrieval, if the LLM component of a RAG chain cannot effectively utilize the retrieved context to generate accurate, coherent, and relevant responses, the final output quality will suffer. Issues with generation quality can arise as hallucinations, inconsistencies, or failure to concisely address the user's query, to name a few.

To identify generation quality issues, you can use the approach outlined in the [Evaluation section](#section-4-evaluation). If evaluation results indicate poor generation quality (e.g., low accuracy, coherence, or relevance scores), you'll need to investigate further to identify the root cause.

The following is a step-by-step process to address **generation quality** issues:

1. Identify a set of test queries with low generation quality metrics.

2. For each query, manually examine the generated response and compare it to the retrieved context and the ground-truth response.

3. Look for patterns or common issues among the queries with low generation quality. Some examples:
   - Generating information not present in the retrieved context or outputting contradicting information with respect to the retrieved context (i.e., hallucination)
   - Failure to directly address the user's query given the provided retrieved context  
   - Generating responses that are overly verbose, difficult to understand or lack logical coherence

4. Based on the identified issues, hypothesize potential root causes and corresponding fixes. See the "[Common reasons for poor generation quality](#common-reasons-for-poor-generation-quality)" table below for guidance.

5. Implement the proposed fix for the most promising or impactful root cause. This may involve modifying the RAG chain (e.g., adjusting the prompt template, trying a different LLM) or the data pipeline (e.g., adjusting the chunking strategy to provide more context).

6. Re-run evals on the updated system and compare generation quality metrics to the previous version. If there is significant improvement, consider deploying the updated RAG application for further testing with end-users (see the [Deployment](#deployment) section).

7. If the generation quality is still not satisfactory, repeat steps 4-6 for the next most promising fix until the desired performance is achieved.

##### Common reasons for poor generation quality

Each of these potential fixes are tagged as one of three types. Based on the type of change, you will follow different steps in section 3.


<table>
<thead>
<tr>
<th>Generation Issue</th>
<th>Debugging Steps</th>
<th>Potential Fix</th>
</tr>
</thead>
<tbody>
<tr>
<td>Generating information not present in the retrieved context (e.g., hallucinations)</td>
<td><ul><li>Compare generated responses to retrieved context to identify hallucinated information</li><li>Assess if certain types of queries or retrieved context are more prone to hallucinations</td>
<td><ul><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Update prompt template to emphasize reliance on retrieved context</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Use a more capable LLM</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Implement a fact-checking or verification step post-generation</td>
</tr>
<tr>
<td>Failure to directly address the user&#39;s query or providing overly generic responses</td>
<td><ul><li>Compare generated responses to user queries to assess relevance and specificity</li><li>Check if certain types of queries result in the correct context being retrieved, but the LLM producing low quality output</td>
<td><ul><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Improve prompt template to encourage direct, specific responses</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Retrieve more targeted context by improving the retrieval process</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Re-rank retrieval results to put most relevant chunks first, only provide these to the LLM</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Use a more capable LLM</td>
</tr>
<tr>
<td>Generating responses that are difficult to understand or lack logical flow</td>
<td><ul><li>Assess output for logical flow, grammatical correctness, and understandability</li><li>Analyze if incoherence occurs more often with certain types of queries or when certain types of context are retrieved</td>
<td><ul><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Change prompt template to encourage coherent, well-structured response</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Provide more context to the LLM by retrieving additional relevant chunks</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Use a more capable LLM</td>
</tr>
<tr>
<td>Generated responses are not in the desired format or style</td>
<td><ul><li>Compare output to expected format and style guidelines</li><li>Assess if certain types of queries or retrieved context are more likely to result in format/style deviations</td>
<td><ul><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Update prompt template to specify the desired output format and style</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Implement a post-processing step to convert the generated response into the desired format</li><li><img src="../_images/chain_code.png" alt="chain-code" height="20"/> Add a step to validate output structure/style, and output a fallback answer if needed.</li><li><img src="../_images/chain_config.png" alt="chain-config" height="20"/> Use an LLM fine-tuned to provide outputs in a specific format or style</td>
</tr>
</tbody>
</table>



### **Step 3:** Implement and evaluate changes

As discussed above, when working to improve the quality of the RAG system, changes can be broadly categorized into three buckets:

1. **![Data pipeline](../images/5-hands-on/data_pipeline.png)** Data pipeline changes
2. **![Chain config](../images/5-hands-on/chain_config.png)** RAG chain configuration changes
3. **![Chain code](../images/5-hands-on/chain_code.png)** RAG chain code changes

Depending on the specific issue you are trying to address, you may need to apply changes to one or both of these components. In some cases, simultaneous changes to both the data pipeline and RAG chain may be necessary to achieve the desired quality improvements.

#### Data pipeline changes

**Data pipeline changes** involve modifying how input data is processed, transformed, or stored before being used by the RAG chain. Examples of data pipeline changes include (and are not limited to):

- Trying a different chunking strategy
- Iterating on the document parsing process
- Changing the embedding model

Implementing a data pipeline change will generally require re-running the entire pipeline to create a new vector index. This process involves reprocessing the input documents, regenerating the vector embeddings, and updating the vector index with new embeddings and metadata.

#### RAG chain changes

**RAG chain changes** involve modifying steps or parameters of the RAG chain itself, without necessarily changing the underlying vector database. Examples of RAG chain changes include (and are not limited to):

- Changing the LLM
- Modifying the prompt template
- Adjusting the retrieval component (e.g., number of retrieval chunks, reranking, query expansion)
- Introducing additional processing steps such as a query understanding step

RAG chain updates may involve editing the **RAG chain configuration file** (e.g., changing the LLM parameters or prompt template), *or* modifying the actual **RAG chain code** (e.g., adding new processing steps or retrieval logic).

#### Testing a potential fix that could improve quality

Once you have identified a potential fix based on the debugging process outlined above, follow these steps to test your changes:

1. Make the necessary changes to the data pipeline or RAG chain code
   - See the [code examples](#code-examples) below for how and where to make these changes
   - If required, re-run the data pipeline to update the vector index with the new embeddings and metadata

2. Log a new version of your chain to MLflow
   - Ensure that any config files (i.e., for both your data pipeline and RAG chain) are logged to the MLflow run
     - <SCREENSHOT OF LOGGED CHAIN>

3. Run evaluation on this new chain

4. Review evaluation results
   - Analyze the evaluation metrics to determine if there has been an improvement the RAG chain's performance
     - <SCREENSHOT OF EVALS>
   - Compare the traces and LLM judge results for individual queries before and after the changes to gain insights into the impact of your changes

5. Iterate on the fixes
   - If the evaluation results do not show the desired improvement, iterate on your changes based on the insights gained from analysis.
   - Repeat steps 1-4 until you are satisfied with the improvement in the RAG chain's output quality

6. Deploy the updated RAG chain for user feedback
   - Once evaluation results indicate improvement, register the chain to Unity Catalog and deploy the updated RAG chain via the Review App.
   - Gather feedback from stakeholders and end-users through one or both of the following:
     - Have stakeholders interact with the app directly in the RAG Studio UI and provide feedback on response quality
       - <SCREENSHOT OF REVIEW APP>
     - Generate responses using the updated chain for the set of evaluation queries and seek feedback on those specific responses

7. Monitor and analyze user feedback
   - Review these results using a [dashboard](https://docs.databricks.com/en/dashboards/index.html#dashboards).
   - Monitor metrics such as the percentage of positive and negative feedback, as well as any specific comments or issues raised by users.

<!--
### Code Examples
| | Component | Change(s) |
|---|---|---|
| **Data pipeline changes**<br><br>1. Re-run data pipeline to create new vector index<br>2. Log new version of RAG chain using the updated index<br>3. Run evals on new chain | [Parser](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/02_parse_docs.py) | - Change parsing strategy<br>  - [Add new parsing strategy to notebook](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/parser_library.py)<br>  - [Update data pipeline config](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/00_config.py#L29) |
| | [Chunking](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/03_chunk_docs.py) | - Chunking strategy<br>  - [Add or update existing chunking strategy](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/chunker_library.py)<br>  - [Update data pipeline config](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/00_config.py#L30-L35)<br>- Change chunk sizes of existing chunking strategy<br>  - [Update data pipeline config](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/00_config.py#L30-L35)<br>- Add metadata to chunks<br>- Semantic chunking |
| | [Embedding<br>model](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/04_vector_index.py#L41) | - Change embedding model<br>  - [Update data pipeline config](https://github.com/databricks-field-eng/field-ai-examples/blob/main/dev/data_processing/notebook_version/data_prep/00_config.py#L18-L24) |
| **RAG chain config changes**<br><br>1. If no changes to data pipeline, do *not* re-run data pipeline<br>2. Log new version of RAG chain using the updated index<br>3. Run evals on new chain | [LLM](#llm) | - Change LLM or its parameters<br>  - [Update RAG chain config](https://github.com/epec254/rag_code/blob/main/RAG%20Cookbook/B_pdf_rag_with_multi_turn_chat/2_rag_chain_config.yaml#L1-L4) |
| | [Prompt<br>Template](/nbs/3-deep-dive.md#prompt-augmentation) | - Iterate on prompt template<br>  - [Update RAG chain config](https://github.com/epec254/rag_code/blob/main/RAG%20Cookbook/B_pdf_rag_with_multi_turn_chat/2_rag_chain_config.yaml#L5-L14) |
| | [Hybrid search](/nbs/3-deep-dive.md#retrieval) | - Try hybrid search instead of semantic search<br>  - Update RAG chain code |
| **RAG chain code changes**<br><br>1. If no changes to data pipeline, *do not* re-run data pipeline<br>2. Log new version of RAG chain using the updated index<br>3. Run evals on new chain | [Reranker](/nbs/3-deep-dive.md#retrieval) | - Add reranker step to RAG chain<br>  - [Update RAG chain code](https://github.com/epec254/rag_code/pull/19) |
| | [Query<br>Expansion](/nbs/3-deep-dive.md#query-understanding) | - Add query expansion step<br>  - Update RAG chain code<br>  - NOTE: Implement this [example prompt](https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook/#query-rewriting-custom) into the multi turn |
| | [Guardrails](/nbs/3-deep-dive.md#post-processing-guardrails) | - Add post-processing guardrails step to RAG chain<br>  - Create a version of this [chain](https://github.com/epec254/rag_code/tree/main/RAG%20Cookbook/B_pdf_rag_with_multi_turn_chat) that includes a sample guardrail prompt using the current advanced DBdemo as an example |
-->
## Deployment

```{image} ../images/5-hands-on/17_img.png
:align: center
```