## Assessing Performance: Metrics that Matter

With an evaluation set, you are able to measure the performance of your RAG application across a number of different dimensions, including:

- **Retrieval quality**: Retrieval metrics assess how successfully your RAG application retrieves relevant supporting data. Precision and recall are two key retrieval metrics.
- **Response quality**: Response quality metrics assess how well the RAG application responds to a user's request. Response metrics can measure, for instance, how well-grounded the response was given the retrieved context, or how harmful/harmless the response was.
- **Chain performance:** Chain metrics capture the overall cost and performance of RAG applications. Overall latency and token consumption are examples of chain performance metrics.

There are two key approaches to measuring performance across these metrics:

- **Ground truth based:** This approach involves comparing the RAG application's retrieved supporting data or final output to the ground-truth answers and supporting documents recorded in the evaluation set. It allows for assessing the performance based on known correct answers.
- **LLM judge based:** In this approach, a separate [LLM acts as a judge](https://arxiv.org/abs/2306.05685) to evaluate the quality of the RAG application's retrieval and responses. LLM judges can be configured to compare the final response to the user query and rate its relevance. This approach automates evaluation across numerous dimensions. LLM judges can also be configured to return rationales for their ratings.

Take time to ensure that the LLM judge's evaluations align with the RAG application's success criteria. Some LLM-as-judge metrics still rely on the ground truth from the evaluation set, which the judge LLM uses to assess the application's output.

### Retrieval metrics

Retrieval metrics help you understand if your retriever is delivering relevant results. Retrieval metrics are largely based on precision and recall.

| Metric Name | Question Answered                       | Details                                                                                                                                                                                      |
|-------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Precision   | Is the retrieved supporting data relevant? | Precision is the proportion of retrieved documents that are actually relevant to the user's request. An LLM judge can be used to assess the relevance of each retrieved chunk to the user's request. |
| Recall      | Did I retrieve most/all of the relevant chunks? | Recall is the proportion of all of the relevant documents that were retrieved. This is a measure of the completeness of the results.                                                              |

In the example below, two out of the three retrieved results were relevant to the user's query, so the precision was 0.66 (2/3). The retrieved docs included two out of a total of four relevant docs, so the recall was 0.5 (2/4).

```{image} ../images/4-evaluation/1_img.png
:align: center
```

See the [appendix](#appendix) on precision and recall for more details.

### Response metrics 

Response metrics assess the quality of the final output. "Quality" has many different dimensions when it comes to assessing LLM outputs, and the range of metrics reflects this.

| Metric Name  | Question Answered                                   | Details                                                                                                                                                                              |
|--------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Correctness  | All things considered, did the LLM give an accurate answer? | Correctness is an LLM-generated metric that assesses whether the LLM's output is correct by comparing it to the ground truth in the evaluation dataset.                                      |
| Groundedness | Is the LLM's response a hallucination or is it grounded to the context? | To measure groundedness, an LLM judge compares the LLM's output to the retrieved supporting data and assesses whether the output reflects the contents of the supporting data or whether it constitutes a hallucination. |
| Harmfulness  | Is the LLM responding safely without any harmful or toxic content? | The Harmfulness measure considers only the RAG application's final response. An LLM judge determines whether the response should be considered harmful or toxic.                              |
| Relevance    | Is the LLM responding to the question asked?        | Relevance is based on the user's request and the RAG application's output. An LLM judge provides a rating of how relevant the output is to the response.                             |

It is very important to collect both response and retrieval metrics. A RAG application can respond poorly in spite of retrieving the correct context; it can also provide good responses on the basis of faulty retrievals. Only by measuring both components can we accurately diagnose and address issues in the application.

### Chain metrics

Chain metrics access the overall performance of the whole RAG chain. Cost and latency can be just as important as quality when it comes to evaluating RAG applications. It is important to consider cost and latency requirements early in the process of developing a RAG application as these considerations can affect every part of the application, including both the retrieval method and the LLM used for generation.

| Metric Name | Question Answered                        | Details                                                                                                                                                                                                                         |
|-------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Total tokens | What is the cost of executing the RAG chain? | Token consumption can be used to approximate cost. This metric counts the tokens used across all LLM generation calls in the RAG pipeline. Generally speaking, more tokens lead to higher costs, and finding ways to reduce tokens can reduce costs. |
| Latency     | What is the latency of executing the RAG chain? | Latency measures the time it takes for the application to return a response after the user sends a request. This includes the time it takes the retriever to retrieve relevant supporting data and for the LLM to generate output.               |


## Appendix

### Precision and Recall

Retrieval metrics are based on the concept of Precision & Recall.  Below is a quick primer on Precision & Recall adapted from the excellent [Wikipedia article](https://en.wikipedia.org/wiki/Precision_and_recall).

Precision measures “Of the items* I retrieved, what % of these items are actually relevant to my user’s query? Computing precision does NOT require your ground truth to contain ALL relevant items.

```{image} ../images/4-evaluation/2_img.png
:align: center
:width: 400px
```

Recall measures “Of ALL the items* that I know are relevant to my user’s query, what % did I retrieve?”  Computing recall requires your ground-truth to contain ALL relevant items.

```{image} ../images/4-evaluation/3_img.png
:align: center
:width: 400px
```

\* Items can either be a document or a chunk of a document. 
