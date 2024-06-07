
# Retrieval Augmented Generation
This repo provides production-ready code to build a **high-quality RAG application** using Databricks.  This repo complements the  [Mosaic Generative AI Cookbook]() which provides:
  - A conceptual overview and deep dive into various Generative AI design patterns, such as Prompt Engineering, Agents, RAG, and Fine Tuning
  - An overview of Evaluation-Driven development
  - The theory of every parameter/knob that impacts quality
  - How to root cause quality issues and detemermine which knobs are relevant to experiment with for your use case
  - Best practices for how to experiment with each knob

This repo is intended for use with the Databricks platform.  Specifically:
- [Mosaic AI Agent Framework]() which provides a fast developer workflow with enterprise-ready LLMops & governance
- [Mosaic AI Agent Evaluation]() which provides reliable, quality measurement using proprietary AI-assisted LLM judges to measure quality metrics that are powered by human feedback collected through an intuitive web-based chat UI

![Alt text](./dbxquality.png)

# Table of contents

- [Evaluation-driven development overview](#evaluation-driven-development)
- [How to build a high-quality RAG application](#how-to-build-and-iteratively-improve-a-high-quality-RAG-application)

# Limitations

- To use these notebooks, you need a Single User cluster running DBR 14.3+.

## Evaluation-driven development

If quality is important to your business, evaluation-driven development is Databricks recommended development workflow for building, testing, and deploying RAG applications.  Evaluation-driven development makes alignment with business stakeholders straightforward since it allows you to confidently state, *“we know our application answers the most critical questions to our business correctly and doesn’t hallucinate.”*

This repo follows the evaluation-driven development paradigm.  For a deep dive into Evaluation Driven development, read the [cookbook section]().

## How to build and iteratively improve a high-quality RAG application

Before getting started, follow this [guide]() to work with your business stakeholders to determine your application's requirements.

### Step 1: Deploying a POC to collect stakeholder feedback

To get started, deploy a *quality-ready POC* to stakeholders.  A quality-ready POC is a RAG application that:
- Stakeholders can provide 👍 / 👎 feedback and ✏️ correct answers using a chat-based Web UI 
- Attaches a detailed MLflow Trace to every piece of feedback
- Chain & data preparation code are parameterized and ready for fast, iterative dev loops to improve quality

1. Clone this repo into your Databricks workspace
2. Adjust the `00_global_config` to point to your Unity Catalog schema and data sources.
2. Select a folder within `01_POC_app` folder that corresponds to the data you are using

*Each POC app is configured with reccomended default settings*

| File type                        | Source            | POC application folder |
|----------------------------------|-------------------|------------------------|
| PDF (`.pdf`) files                        |   UC Volume                |        [Single-turn chat]()        \|  [Multi-turn chat]()        |
| PowerPoint (`.pptx`) files                 |       UC Volume            |         [Single-turn chat]()        \|  [Multi-turn chat]()        |
| Word (`.docx`) files                       |    UC Volume               |         [Single-turn chat]()        \|  [Multi-turn chat]()        |
| HTML (`.html`)files                     |    UC Volume               |               [Single-turn chat]()        \|  [Multi-turn chat]()        |
<!-- | HTML text                     |    Delta Table               |               [Single-turn chat]()        \|  [Multi-turn chat]()        | -->
<!-- | Markdown or regular text         |        Delta Table           |            [Single-turn chat]()        \|  [Multi-turn chat]()        | -->
<!-- | JSON files        |         UC Volume          |          [Single-turn chat]()        \|  [Multi-turn chat]()        | -->

> *In contrast to the "3 lines of magic code" POC, there is necessarily more complexity in the code base. However, we have organized this complexity to make quality iteration easy.*

3. Review the POC's parameters in the `00_config` Notebook
4. Run `02_validate_config` to ensure that all necessary resources are set up
4. Run `02_poc_data_pipeline` to create a Vector Index from your data using Databricks Vector Search
5. Run `03_deploy_poc_to_review_app` to deploy the app to a [Quality Lab Review App]()
6. Share the app's URL your stakeholders so they can chat with it and provide feedback

**IMAGE OF REVIEW APP**

## Step 2: Evaluate the POC’s quality

Once your stakeholders have used your POC, we will use their feedback to measure the POC’s quality and establish a baseline.

1. Open the `04_evaluate_poc_quality` Notebook.
2. Follow the steps in the Notebook to create an initial Evaluation Set based on the logs from your Review App.
3. Inspect the Evaluation Set to understand the data that is included.  You need to validate that your Evaluation Set contains a representative and challenging set of questions..
4. Save your evaluation set to a Delta Table for later use
5. Evaluate the POC with Agent Evaluation's LLM Judge-based evaluation.  Open MLflow to view the results.

## Step 3: Iteratively diagnose and fix quality issues

While a basic RAG chain is relatively straightforward to implement, refining it to consistently produce high-quality outputs is often non-trivial. Identifying the root causes of issues and determining which levers of the solution to pull to improve output quality requires understanding the various components and their interactions.

Follow the [omprove RAG quality]() section of the cookbook to:
1. Understand RAG quality improvement levers
2. Identify the root cause of quality issues

Once you have identified the root causes, follow the steps in [02_experimentation/README.md](02_experimentation/README.md) to implement and evaluate experiments.

## Step 4: Deploy to production

With Databricks, this part is easy.  Once you have identified an application configuration in the previous step that meets your production quality bar, simply:
1. Register the chain from your MLflow Experiment to Unity Catalog
2. Use the `databricks.agents.deploy_model(...)` SDK to create a scalable, production-ready REST API in 1-line of code.
