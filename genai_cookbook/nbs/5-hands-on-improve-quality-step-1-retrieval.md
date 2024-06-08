#### Retrieval quality

##### How to debug retrieval quality

**You are on this page because your [root cause analysis](./5-hands-on-improve-quality-step-1.md) said that retrieval quality was the issue to focus.**

Retrieval quality is arguably the most important component of a RAG application. If the most relevant chunks are not returned for a given query, the LLM will not have access to the necessary information to generate a high-quality response. Poor retrieval can thus lead to irrelevant, incomplete, or hallucinated output.  This step requires manual effort to analyze the underlying data. With Mosaic AI, this becomes considerably easier given the tight integration between the data platform (Unity Catalog and Vector Search), and experiment tracking (MLflow LLM evaluation and MLflow tracing).

Here's a step-by-step process to address **retrieval quality** issues:

1. Load the queries from your root cause analysis that had low retrieval quality metrics.

2. For each query, manually examine the retrieved chunks.  If available, compare them to the ground-truth retrieval documents.

3. Look for patterns or common issues among the queries with low retrieval quality. Some examples might include:
   - Relevant information is missing from the vector database entirely
   - Insufficient number of chunks/documents returned for a retrieval query
   - Chunks are too small and lack sufficient context
   - Chunks are too large and contain multiple, unrelated topics
   - The embedding model fails to capture semantic similarity for domain-specific terms

4. Based on the identified issue, hypothesize potential root causes and corresponding fixes. See the [Common reasons for poor retrieval quality](#common-reasons-for-poor-retrieval-quality) table below for guidance on this.

5. Implement the proposed fix for the most promising or impactful root cause by following [step 4.2](./5-hands-on-improve-quality-step-2.md). This may involve modifying the data pipeline (e.g., adjusting chunk size, trying a different embedding model) or modifying the RAG chain (e.g., implementing hybrid search, retrieving more chunks).

6. Re-run the evaluation on the updated system and compare the retrieval quality metrics to the previous version. Once retrieval quality is at a desired level, re-run the [root cause analysis](./5-hands-on-improve-quality-step-1.md) to determine if the overall chain has any additional issues that should be addressed.

7. If retrieval quality is still not satisfactory, repeat steps 4-6 for the next most promising fixes until the desired performance is achieved.

##### Common reasons for poor retrieval quality

Each of these potential fixes are tagged as one of three types. Based on the type of change, you will follow different steps in [step 4.2](./5-hands-on-improve-quality-step-2.md).

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

<br/>
<br/>