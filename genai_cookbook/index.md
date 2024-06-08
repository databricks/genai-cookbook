---
title: Databricks Generative AI Cookbook
---

# Databricks Mosaic Generative AI Cookbook

The Databricks Generative AI Cookbook is a definitive how-to guide for building *high-quality* generative AI applications. *High-quality* applications are:
1. **Accurate:** provides correct responses
2. **Safe:** does not deliver harmful or insecure responses
3. **Governed:** respects permissions & access controls

Developed in partnership with Mosaic AI's research team, this cookbook lays out Databricks best-practice development workflow for building high-quality RAG apps: *evaluation driven development.* It outlines the most relevant knobs & approaches that can increase quality and provides a comprehensive repository of sample code implementing those techniques. This code & cookbook will take you from initial POC to high-quality production-ready application.

> This first release focuses on retrieval-augmented generation (RAG).  Future releases will include the other popular generative AI techniques: agents & function calling, prompt engineering, fine tuning, and pre-training.

## Retrieval-augmented generation (RAG)

The RAG cookbook is divided into 2 sections:
1. [**Learn:**](#learn) Understand the required components of a production-ready, high-quality RAG application
2. [**Implement:**](#implement) Use our sample code to follow the Databricks-recommended developer workflow for delivering a high-quality RAG application


#### Learn

**Table of contents**
1. [RAG overview](./nbs/1-introduction-to-rag): High level overview of the basic concepts of RAG
2. [RAG fundamentals](./nbs/2-fundamentals-unstructured): Introduction to the key components of a RAG application
3. [RAG quality knobs](./nbs/3-deep-dive): Explains the knobs that Databricks recommends tuning in order to improve RAG application quality
4. [RAG quality evaluation deep dive](./nbs/4-evaluation): Understand how RAG evaluation works, including creating evaluation sets, the quality metrics that matter, and required developer tooling
5. [RAG development workflow](nbs/5-rag-development-workflow.md): Understand Databricks recommended development workflow for building, testing, and deploying a high-quality RAG application: evaluation-driven development

**Getting started**

| Time required | Outcome | Link |
|------ | ---- | ---- |
| 🕧 <br/> 10 minutes | Sample RAG app deployed to web-based chat app that collects feedback | [RAG Demo]((https://DBDEMO)) |
| 🕧🕧🕧 <br/>60 minutes | POC RAG app with *your data* deployed to a chat UI that can collect feedback from your business stakeholders | [Build a POC](./nbs/5-hands-on-build-poc.md)|
| 🕧🕧 <br/>30 minutes | Comprehensive quality/cost/latency evaluation of your POC app | [Evaluate your POC](./nbs/5-hands-on-evaluate-poc.md) |


#### Implement

**Table of contents**


1. [Gather Requirements](./nbs/5-hands-on-requirements.md): Requirements you must discover from stakeholders before building a RAG app
2. [Deploy POC to Collect Stakeholder Feedback](./nbs/5-hands-on-build-poc.md): Launch a proof of concept (POC) to gather feedback from stakeholders and understand baseline quality
3. [Evaluate POC’s Quality](./nbs/5-hands-on-evaluate-poc.md): Assess the quality of your POC to identify areas for improvement
4. [Root Cause & Iteratively Fix Quality Issues](./nbs/5-hands-on-improve-quality.md): Diagnose the root causes of any quality issues and apply iterative fixes to improve the app's quality
5. [Deploy & Monitor](./nbs/5-hands-on-deploy-and-monitor.md): Deploy the finalized RAG app to production and continuously monitor its performance to ensure sustained quality.

**Getting started**


| Time required | Outcome | Link |
|------ | ---- |  ---- |
| 🕧 <br/> 5 minutes | Understand how RAG works at a high-level | [Intro to RAG](./nbs/1-introduction-to-rag.md) |
| 🕧🕧 <br/> 30 minutes  |Understand the key components in a RAG app |  [RAG fundamentals](./nbs/2-fundamentals-unstructured.md) |
| 🕧🕧🕧 <br/> 60 minutes | Understand the knobs Databricks recommends tuning improve RAG app quality | [RAG quality knobs](./nbs/3-deep-dive.md) |
