# Evaluating RAG quality


The old saying "you can't manage what you can't measure" is incredibly relevant (no pun intended) in the context of any generative AI application, RAG included. In order for your generative AI application to deliver high quality, accurate responses, you **must** be able to define and measure what "quality" means for your use case.

This section deep dives into 3 critical components of evaluation:

1. [Establishing Ground Truth: Creating Evaluation Sets](#establishing-ground-truth-creating-evaluation-sets)
2. [Assessing Performance: Defining Metrics that Matter](#assessing-performance-defining-metrics-that-matter)  
3. [Enabling Measurement: Building Supporting Infrastructure](#enabling-measurement-building-supporting-infrastructure)

## Establishing Ground Truth: Creating Evaluation Sets

To measure quality, Databricks recommends creating a human-labeled Evaluation Set, which is a curated, representative set of queries, along with ground-truth answers and (optionally) the correct supporting documents that should be retrieved. Human input is crucial in this process, as it ensures that the Evaluation Set accurately reflects the expectations and requirements of the end-users.

A good Evaluation Set has the following characteristics:

- **Representative:** Accurately reflects the variety of requests the application will encounter in production.
- **Challenging:** The set should include difficult and diverse cases to effectively test the model's capabilities.  Ideally, it will include adversarial examples such as questions attempting prompt injection or questions attempting to generate inappropriate responses from LLM.
- **Continually updated:** The set must be periodically updated to reflect how the application is used in production and the changing nature of the indexed data.

Databricks recommends at least 30 questions in your evaluation set, and ideally 100 - 200. The best evaluation sets will grow over time to contain 1,000s of questions.

To avoid overfitting, Databricks recommends splitting your evaluation set into training, test, and validation sets:

- Training set: ~70% of the questions. Used for an initial pass to evaluate every experiment to identify the highest potential ones.
- Test set: ~20% of the questions. Used for evaluating the highest performing experiments from the training set.  
- Validation set: ~10% of the questions. Used for a final validation check before deploying an experiment to production.

## Assessing Performance: Defining Metrics that Matter

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

## Enabling Measurement: Building Supporting Infrastructure

Measuring quality is not easy and requires a significant infrastructure investment. This section details what you need to succeed and how Databricks provides these components.

**Detailed trace logging.** The core of your RAG application's logic is a series of steps in the chain. In order to evaluate and debug quality, you need to implement instrumentation that tracks the chain's inputs and outputs, along with each step of the chain and its associated inputs and outputs. The instrumentation you put in place should work the same way in development and production.

In Databricks, MLflow Trace Logging provides this capability. With MLflow Trace Logging, you instrument your code in production, and get the same traces during development and in production. For more details, [link the docs].

**Stakeholder review UI.** Most often, as a developer, you are not a domain expert in the content of the application you are developing. In order to collect feedback from human experts who can assess your application's output quality, you need an interface that allows them to interact with early versions of the application and provide detailed feedback. Further, you need a way to load specific application outputs for the stakeholders to assess their quality.

This interface must track the application's outputs and associated feedback in a structured manner, storing the full application trace and detailed feedback in a data table.

In Databricks, the Quality Lab Review App provides this capability. For more details, [link the docs].

**Quality / cost / latency metric framework.** You need a way to define the metrics that comprehensively measure the quality of each component of your chain and the end-to-end application. Ideally, the framework would provide a suite of standard metrics out of the box, in addition to supporting customization, so you can add metrics that test specific aspects of quality that are unique to your business.

**Evaluation harness.** You need a way to quickly and efficiently get outputs from your chain for every question in your evaluation set, and then evaluate each output on the relevant metrics. This harness must be as efficient as possible, since you will run evaluation after every experiment that you try to improve quality.

In Databricks, Quality Lab provides these capabilities. [link more details]

**Evaluation set management.** Your evaluation set is a living, breathing set of questions that you will update iteratively over the course of your application's development and production lifecycle.

**Experiment tracking framework.** During the course of your application development, you will try many different experiments. An experiment tracking framework enables you to log each experiment and track its metrics vs. other experiments.

**Chain parameterization framework.** Many experiments you try will require you to hold the chain's code constant while iterating on various parameters used by the code. You need a framework that enables you to do this.

In Databricks, MLflow provides these capabilities. [link]

Online monitoring. [talk about LHM]


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
