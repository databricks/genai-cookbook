# Databricks Generative AI Cookbook

Please visit [ai-cookbook.io](http://ai-cookbook.io) for the accompanying documentation for this repository.

This repository provides [learning materials](https://ai-cookbook.io/) and code examples to build a **high-quality Generative AI application** using Databricks. The Cookbook provides:

- A conceptual overview and deep dive into various Generative AI design patterns, such as Prompt Engineering, Agents, RAG, and Fine-Tuning.
- An overview of Evaluation-Driven Development.
- The theory of every parameter/knob that impacts quality.
- How to root cause quality issues and determine which knobs are relevant to experiment with for your use case.
- Best practices for how to experiment with each knob.

## TL;DR:

This repository is a monorepo - each directory contains a standalone "recipe". 

Choose the recipe that best matches your needs:

- **For RAG Applications:**
  - [RAG Getting Started](./rag_app_sample_code/README.md)
  - Start with `agent_app_sample_code/A_POC_app` to build a proof of concept
  - Then explore `agent_app_sample_code/B_quality_iteration` to improve quality
  - Uses Databricks Agent Framework + Evaluation for enterprise features
    - Ingest, process and automatically index documents with Spark + Databricks Vector Search
    - Experiment and version RAG models with MLflow and Unity Catalog
    - Autoscaling model deployment with Model Serving
    - Iterate on quality with Agent Evals and SME Review UI
    - Monitor real-time performance

- **For an agent that uses a retriever tool:**
  - Check out `agent_app_sample_code`

- **For Agent Application in pure Python + OpenAI SDK:**
  - [OpenAI SDK Getting Started](./openai_sdk_agent_app_sample_code/README.md)
  - Navigate to `openai_sdk_agent_app_sample_code`
  - Examples of building agents using OpenAI SDK with MLflow PyFunc Models

## How to use this repository

The provided code is intended for use with the Databricks platform.  Specifically:
- [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) which provides a fast developer workflow with enterprise-ready LLMops & governance
- [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) which provides reliable, quality measurement using proprietary AI-assisted LLM judges to measure quality metrics that are powered by human feedback collected through an intuitive web-based chat UI

![Alt text](rag_app_sample_code/dbxquality.png)

Specific instructions for each recipe are provided in the README.md file of each subdirectory.

### Prerequisites

- Your Databricks workspace must have [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html) enabled. 
- Your Databricks workspace must have [Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html#enable-model-serving-for-your-workspace) enabled.

### Option 1: Running in a Databricks Workspace (Recommended)

1. **Clone the Repository into your Databricks Workspace:**
  - In your Databricks workspace, go to Repos
  - Click "Add Repo"
  - Enter the Git repository URL: `https://github.com/databricks/genai-cookbook.git`
  - After completing the steps above, use sparse checkout mode to clone the subdirectory of your choice.

1a. **Optional: Download the repository as a zip file:**
In cases where you cannot use Git folders, you can download the repository as a zip file.
  - Click the "Download ZIP" button on the repository page
  - Unzip the file and upload it to your Databricks workspace

2. **Set Up Your Databricks Environment:**
  - Use Serverless Notebooks or create a new cluster with Databricks Runtime 14.0 or higher

3. **Run the Sample Code:**
  - Navigate to `agent_app_sample_code/A_POC_app` to start with a proof of concept
  - Follow the numbered notebooks in sequence
  - Each notebook contains detailed instructions and explanations

### Option 2: Running Locally

If you prefer to edit code locally, you can use Databricks Connect and optionally an IDE plugin like [VSCode](https://docs.databricks.com/en/dev-tools/vscode-ext/index.html). 

It is strongly recommended to first read the [Databricks Connect documentation](https://docs.databricks.com/en/
dev-tools/databricks-connect/index.html) to understand how to connect your local machine to your Databricks 
workspace.

1. **Install Prerequisites**
   - Python 3.10 or higher
   - [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)
   - Git
   - Optional: [VSCode with Databricks Extension](https://docs.databricks.com/en/dev-tools/vscode-ext/index.html)

2. **Set Up Databricks Connect**
   - Read the [Databricks Connect documentation](https://docs.databricks.com/en/dev-tools/databricks-connect/index.html)
   - Follow the setup instructions to connect your local machine to your workspace

3. **Configure MLflow**
   Set the following environment variables:
   ```bash
   export MLFLOW_TRACKING_URI=databricks
   export DATABRICKS_HOST=<your-workspace-url>
   export DATABRICKS_TOKEN=<your-access-token>
   ```

4. **Clone and Set Up the Repository**
   ```bash
   git clone https://github.com/databricks/genai-cookbook.git
   cd genai-cookbook
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r dev/dev_requirements.txt
   ```

5. **Configure Databricks CLI**
   ```bash
   databricks configure --token
   ```
   When prompted, enter your:
   - Workspace URL
   - Access token

## Contributing

We welcome contributions to improve the cookbook! Here's how you can help:

### Development Setup

1. **Fork and Clone:**
  - Fork the repository
  - Clone the forked repository
   ```bash
   git clone https://github.com/YOUR_USERNAME/genai-cookbook.git
   cd genai-cookbook
   ```

### Making Changes

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Update Documentation:**
   - If you're adding new a new coobook directory, add a README.md file to the directory describing the new cookbook.

3. **Code Style:**
   - Follow PEP 8 guidelines
   - Include docstrings for new functions
   - Add type hints where possible

### Submitting Changes

1. **Commit Your Changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

2. **Push to Your Fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request:**
   - Go to the [Pull Requests](https://github.com/databricks/genai-cookbook/pulls) page
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template with:
     - Description of changes
     - Related issues
     - Testing performed
     - Screenshots/videos of manual testing (required)

### Getting Help

- For bugs or feature requests, or questions, [create an issue](https://github.com/databricks/genai-cookbook/issues)
