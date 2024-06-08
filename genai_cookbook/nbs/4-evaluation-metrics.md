## Assessing performance: Metrics that Matter

With an evaluation set, you are able to measure the performance of your RAG application across a number of different dimensions, including:

- **Retrieval quality**: Retrieval metrics assess how successfully your RAG application retrieves relevant supporting data. Precision and recall are two key retrieval metrics.
- **Response quality**: Response quality metrics assess how well the RAG application responds to a user's request. Response metrics can measure, for instance, if the resulting answer is accurate per the ground-truth, how well-grounded the response was given the retrieved context (e.g., did the LLM hallucinate), or how safe the response was (e.g., no toxicity).
- **Cost & latency:** Chain metrics capture the overall cost and performance of RAG applications. Overall latency and token consumption are examples of chain performance metrics.

It is very important to collect both response and retrieval metrics. A RAG application can respond poorly in spite of retrieving the correct context; it can also provide good responses on the basis of faulty retrievals. Only by measuring both components can we accurately diagnose and address issues in the application.

There are two key approaches to measuring performance across these metrics:

- **Deterministic measurement:** Cost and latency metrics can be computed deterministically based on the application's outputs.  If your evaluation set includes a list of documents that contain the answer to a question, a subset of the retrieval metrics can also be computed deterministically.
- **LLM judge based measurement** In this approach, a separate [LLM acts as a judge](https://arxiv.org/abs/2306.05685) to evaluate the quality of the RAG application's retrieval and responses.  Some LLM judges, such as answer correctness, compare the human-labeled ground truth vs. the app's outputs.  Other LLM judges, such as groundedness, do not require human-labeled ground truth to assess their app's outputs.

Take time to ensure that the LLM judge's evaluations align with the RAG application's success criteria.

> [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) provides an out-of-the-box implementation, using hosted LLM judge models, for each metric discussed on this page.  Quality Lab's documentation discusses the [details](https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html) of how these metrics and judges are implemented and provides [capabilities](https://docs.databricks.com/generative-ai/agent-evaluation/advanced-agent-eval.html#provide-examples-to-the-built-in-llm-judges) to tune the judge's with your data to increase their accuracy

### Metric overview

Below is a summary of the metrics that Databricks recommends for measuring the quality, cost, and latency of your RAG application.  These metrics are implemented in [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html).

<table class="table">
<thead>
<tr>
<th style="width: 10%;">Dimension</th>
<th style="width: 15%;">Metric name</th>
<th style="width: 60%;">Question</th>
<th style="width: 10%;">Measured by</th>
<th style="width: 5%;">Ground truth required?</th>

</tr>
</thead>
<tbody>
<tr>
<td>Retrieval</td>
<td><code>chunk_relevance/precision</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#chunk-relevance-precision">What % of the retrieved chunks are relevant to the request?</a></td>
<td>LLM judge</td>
<td>✖️</td>
</tr>
<tr>
<td>Retrieval</td>
<td><code>document_recall</code></td>
<td><a href="TODO">What % of the ground truth documents are represented in the retrieved chunks?</a></td>
<td>Deterministic</td>
<td>✔️</td>
</tr>
<tr>
<td>Response</td>
<td><code>correctness</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#correctness">Overall, did the agent generate a correct response?</a></td>
<td>LLM judge</td>
<td>✔️</td>
</tr>
<tr>
<td>Response</td>
<td><code>relevance_to_query</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#answer-relevance">Is the response relevant to the request?</a></td>
<td>LLM judge</td>
<td>✖️</td>
</tr>
<tr>
<td>Response</td>
<td><code>groundedness</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#groundedness">Is the response a hallucination or grounded in context?</a></td>
<td>LLM judge</td>
<td>✖️</td>
</tr>
<tr>
<td>Response</td>
<td><code>safety</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#safety">Is there harmful content in the response?</a></td>
<td>LLM judge</td>
<td>✖️</td>
</tr>
<tr>
<td>Cost</td>
<td><code>total_token_count</code>, <code>total_input_token_count</code>, <code>total_output_token_count</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#token-count">What&#39;s the total count of tokens for LLM generations?</a></td>
<td>Deterministic</td>
<td>✖️</td>
</tr>
<tr>
<td>Latency</td>
<td><code>latency_seconds</code></td>
<td><a href="https://docs.databricks.com/generative-ai/agent-evaluation/llm-judge-metrics.html#latency">What&#39;s the latency of executing the app?</a></td>
<td>Deterministic</td>
<td>✖️</td>
</tr>
</tbody>
</table>


#### Understanding how retrieval metrics work

Retrieval metrics help you understand if your retriever is delivering relevant results. Retrieval metrics are based on precision and recall.

| Metric Name | Question Answered                       | Details                                                                                                                                                                                      |
|-------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Precision   | What % of the retrieved chunks are relevant to the request? | Precision is the proportion of retrieved documents that are actually relevant to the user's request. An LLM judge can be used to assess the relevance of each retrieved chunk to the user's request. |
| Recall      | What % of the ground truth documents are represented in the retrieved chunks? | Recall is the proportion of the ground truth documents that are represented in the retrieved chunks. This is a measure of the completeness of the results.                                                              |

Below is a quick primer on Precision & Recall adapted from the excellent [Wikipedia article](https://en.wikipedia.org/wiki/Precision_and_recall).

Precision measures *"Of the chunks I retrieved, what % of these items are actually relevant to my user’s query?"* Computing precision does NOT require knowing ALL relevant items.

```{image} ../images/4-evaluation/2_img.png
:align: center
:width: 400px
```

Recall measures *"Of ALL the document that I know are relevant to my user’s query, what % did I retrieve a chunk from?"*  Computing recall requires your ground-truth to contain ALL relevant items.

```{image} ../images/4-evaluation/3_img.png
:align: center
:width: 400px
```

\* Items can either be a document or a chunk of a document. 

In the example below, two out of the three retrieved results were relevant to the user's query, so the precision was 0.66 (2/3). The retrieved docs included two out of a total of four relevant docs, so the recall was 0.5 (2/4).

```{image} ../images/4-evaluation/1_img.png
:align: center
```