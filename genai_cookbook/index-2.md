---
title: Databricks GenAI Cookbook
---

# GenAI Cookbook
**TLDR;** this cookbook and its sample code will take you from initial POC to high-quality production-ready application using [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) and [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) on the Databricks platform.

The Databricks Generative AI Cookbook is a definitive how-to guide for building *high-quality* generative AI applications. *High-quality* applications are applications that:
1. **Accurate:** provide correct responses
2. **Safe:** do not deliver harmful or insecure responses
3. **Governed:** respect data permissions & access controls and track lineage

Developed in partnership with Mosaic AI's research team, this cookbook lays out Databricks best-practice development workflow for building high-quality RAG apps: *evaluation driven development.* It outlines the most relevant knobs & approaches that can increase RAG application quality and provides a comprehensive repository of sample code implementing those techniques. 

```{important}
- Only have a few minutes and want to see a demo of Mosaic AI Agent Framework & Agent Evaluation?  Start [here](https://ai-cookbook.io/10-min-demo/mosaic-ai-agents-demo-dbx-notebook.html).
- Want to hop into code and deploy a RAG POC using your data?  Start [here](./nbs/6-implement-overview.md).
- Don't have any data, but want to deploy a sample RAG application?  Start here.
```

```{image} images/index/dbxquality.png
:align: center
```

<br/>


```{image} images/5-hands-on/review_app2.gif
:align: center
```

<br/>

This cookbook is intended for use with the Databricks platform.  Specifically:
- [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/retrieval-augmented-generation.html) which provides a fast developer workflow with enterprise-ready LLMops & governance
- [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) which provides reliable, quality measurement using proprietary AI-assisted LLM judges to measure quality metrics that are powered by human feedback collected through an intuitive web-based chat UI


# Retrieval-augmented generation (RAG)

> This first release focuses on retrieval-augmented generation (RAG).  Future releases will include the other popular generative AI techniques: agents & function calling, prompt engineering, fine tuning, and pre-training.

The RAG cookbook is divided into 2 sections:
1. **Learn:** Understand the required components of a production-ready, high-quality RAG application
2. **Implement:** Use our sample code to follow an evaluation-driven workflow for delivering a high-quality RAG application

## Code-based quick starts

| Time required | Outcome | Link |
|------ | ---- | ---- |
| 🕧 <br/> 30 minutes | Sample RAG app deployed to web-based chat app that collects feedback | [RAG Demo](https://ai-cookbook.io/10-min-demo/mosaic-ai-agents-demo-dbx-notebook.html) |
| 🕧🕧🕧 <br/>2 hours | POC RAG app with *your data* deployed to a chat UI that can collect feedback from your business stakeholders | [Build & deploy a POC](./nbs/5-hands-on-build-poc.md)|
| 🕧🕧 <br/>1 hour | Comprehensive quality/cost/latency evaluation of your POC app | - [Evaluate your POC](./nbs/5-hands-on-evaluate-poc.md) <br/> - [Identify the root causes of quality issues](./nbs/5-hands-on-improve-quality-step-1.md) |



## Table of contents
<!--
**Table of contents**
1. [RAG overview](./nbs/1-introduction-to-rag): Understand how RAG works at a high-level
2. [RAG fundamentals](./nbs/2-fundamentals-unstructured): Understand the key components in a RAG app
3. [RAG quality knobs](./nbs/3-deep-dive): Understand the knobs Databricks recommends tuning improve RAG app quality 
4. [RAG quality evaluation deep dive](./nbs/4-evaluation): Understand how RAG evaluation works, including creating evaluation sets, the quality metrics that matter, and required developer tooling
5. [Evaluation-driven development](nbs/5-rag-development-workflow.md): Understand Databricks recommended development workflow for building, testing, and deploying a high-quality RAG application: evaluation-driven development-->

```{tableofcontents}
```
<!--
#### Implement

**Table of contents**


1. [Gather Requirements](./nbs/5-hands-on-requirements.md): Requirements you must discover from stakeholders before building a RAG app
2. [Deploy POC to Collect Stakeholder Feedback](./nbs/5-hands-on-build-poc.md): Launch a proof of concept (POC) to gather feedback from stakeholders and understand baseline quality
3. [Evaluate POC’s Quality](./nbs/5-hands-on-evaluate-poc.md): Assess the quality of your POC to identify areas for improvement
4. [Root Cause & Iteratively Fix Quality Issues](./nbs/5-hands-on-improve-quality.md): Diagnose the root causes of any quality issues and apply iterative fixes to improve the app's quality
5. [Deploy & Monitor](./nbs/5-hands-on-deploy-and-monitor.md): Deploy the finalized RAG app to production and continuously monitor its performance to ensure sustained quality.
-->
