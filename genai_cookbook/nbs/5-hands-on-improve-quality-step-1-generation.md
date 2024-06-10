#### Debugging generation quality

##### Debugging generation quality

**You are on this page because your [root cause analysis](./5-hands-on-improve-quality-step-1.md) said that improving LLM generation was a root cause to address.**

Even with optimal retrieval, if the LLM component of a RAG chain cannot effectively utilize the retrieved context to generate accurate, coherent, and relevant responses, the final output quality will suffer. Issues with generation quality can arise as hallucinations, inconsistencies, or failure to concisely address the user's query, to name a few.

The following is a step-by-step process to address **generation quality** issues:



1. Open the `quality_iteration/01_root_cause_quality_issues` Notebook

2. Use the queries to load MLflow traces of the records that retrieval quality issues.

2. For each record, manually examine the generated response and compare it to the retrieved context and the ground-truth response.

3. Look for patterns or common issues among the queries with low generation quality. Some examples:
   - Generating information not present in the retrieved context or outputting contradicting information with respect to the retrieved context (i.e., hallucination)
   - Failure to directly address the user's query given the provided retrieved context  
   - Generating responses that are overly verbose, difficult to understand or lack logical coherence

4. Based on the identified issues, hypothesize potential root causes and corresponding fixes. See the "[Common reasons for poor generation quality](#common-reasons-for-poor-generation-quality)" table below for guidance.

5. Follow the steps in [implement and evaluate changes](./5-hands-on-improve-quality-step-2.md) to implement and evaluate a potential fix.
      - This may involve modifying the RAG chain (e.g., adjusting the prompt template, trying a different LLM) or the data pipeline (e.g., adjusting the chunking strategy to provide more context).

7. If the generation quality is still not satisfactory, repeat steps 4 - 5 for the next most promising fix until the desired performance is achieved.

7. Re-run the [root cause analysis](./5-hands-on-improve-quality-step-1.md) to determine if the overall chain has any additional root causes that should be addressed.


##### Common reasons for poor generation quality

Each of these potential fixes are can be broadly categorized into three buckets:

1. **![Data pipeline](../images/5-hands-on/data_pipeline.png)** changes
2. **![Chain config](../images/5-hands-on/chain_config.png)** changes
3. **![Chain code](../images/5-hands-on/chain_code.png)** changes

Based on the type of change, you will follow different steps in the [implement and evaluate changes](./5-hands-on-improve-quality-step-2.md) step.
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

<br/>
<br/>