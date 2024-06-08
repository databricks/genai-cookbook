## Enabling Measurement: Supporting Infrastructure

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
