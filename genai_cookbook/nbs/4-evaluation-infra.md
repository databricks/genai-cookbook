## Enabling Measurement: Supporting Infrastructure

Measuring quality is not easy and requires a significant infrastructure investment. This section details what you need to succeed and how Databricks provides these components.

**Detailed trace logging.** The core of your RAG application's logic is a series of steps in the chain. In order to evaluate and debug quality, you need to implement instrumentation that tracks the chain's inputs and outputs, along with each step of the chain and its associated inputs and outputs. The instrumentation you put in place should work the same way in development and production.

> In Databricks, [MLflow Tracing](https://docs.databricks.com/mlflow/mlflow-tracing.html) provides this capability. With MLflow Trace Logging, you instrument your code in production, and get the same traces during development and in production.  Production traces are logged as part of the [Inference Table](https://docs.databricks.com/machine-learning/model-serving/inference-tables.html).

**Stakeholder review UI.** Most often, as a developer, you are not a domain expert in the content of the application you are developing. In order to collect feedback from human experts who can assess your application's output quality, you need an interface that allows them to interact with early versions of the application and provide detailed feedback. Further, you need a way to load specific application outputs for the stakeholders to assess their quality.

This interface must track the application's outputs and associated feedback in a structured manner, storing the full application trace and detailed feedback in a data table.

> In Databricks, the [Quality Lab Review App](https://docs.databricks.com/generative-ai/agent-evaluation/human-evaluation.html) provides this capability. 

**Quality / cost / latency metric framework.** You need a way to define the metrics that comprehensively measure the quality of each component of your chain and the end-to-end application. Ideally, the framework would provide a suite of standard metrics out of the box, in addition to supporting customization, so you can add metrics that test specific aspects of quality that are unique to your business.

> In Databricks, [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) provides an out-of-the-box implementation, using hosted LLM judge models, for the necessary quality/cost/latency metrics.

**Evaluation harness.** You need a way to quickly and efficiently get outputs from your chain for every question in your evaluation set, and then evaluate each output on the relevant metrics. This harness must be as efficient as possible, since you will run evaluation after every experiment that you try to improve quality.

> In Databricks, [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) provides an [evaluation harness](https://docs.databricks.com/generative-ai/agent-evaluation/evaluate-agent.html) that is integrated with MLflow.

**Evaluation set management.** Your evaluation set is a living, breathing set of questions that you will update iteratively over the course of your application's development and production lifecycle.

> In Databricks, you can manage your evaluation set as a Delta Table.  When evaluating with MLflow, MLflow will automatically log a snapshot of the version of the evaluation set used.

**Experiment tracking framework.** During the course of your application development, you will try many different experiments. An experiment tracking framework enables you to log each experiment and track its metrics vs. other experiments.

> In Databricks, [MLflow](https://docs.databricks.com/mlflow/index.html) provides experiment tracking capabilities.

**Chain parameterization framework.** Many experiments you try will require you to hold the chain's code constant while iterating on various parameters used by the code. You need a framework that enables you to do this.

In Databricks, [MLflow model configuration](https://docs.databricks.com/generative-ai/create-log-agent.html#use-parameters-to-control-agent-execution) provides these capabilities. 

**Online monitoring.** Once deployed, you need a way monitor the application's health and on-going quality/cost/latency.

> In Databricks, Model Serving provides [application health monitoring](https://docs.databricks.com/machine-learning/model-serving/monitor-diagnose-endpoints.html) and [Lakehouse Monitoring](https://docs.databricks.com/en/lakehouse-monitoring/index.html) provides on-going to dashboard and monitor quality/cost/latency.
