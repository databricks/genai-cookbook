# Databricks Generative AI Cookbook

Please visit [ai-cookbook.io](http://ai-cookbook.io) for the accompanying documentation for this repository.

This repository provides [learning materials](https://ai-cookbook.io/) and code examples to build a **high-quality Generative AI application** using Databricks. The Cookbook provides:

- A conceptual overview and deep dive into various Generative AI design patterns, such as Prompt Engineering, Agents, RAG, and Fine-Tuning.
- An overview of Evaluation-Driven Development.
- The theory of every parameter/knob that impacts quality.
- How to root cause quality issues and determine which knobs are relevant to experiment with for your use case.
- Best practices for how to experiment with each knob.

## TL;DR:

Choose the recipe that best matches your needs:

- **For RAG Applications:**
  - [RAG Getting Started](./rag_app_sample_code/README.md)
  - Start with `agent_app_sample_code/A_POC_app` to build a proof of concept
  - Then explore `agent_app_sample_code/B_quality_iteration` to improve quality
  - Uses Databricks' Mosaic AI Agent Framework for enterprise features

- **For an agent that uses a retriever tool:**
  - Check out `agent_app_sample_code`

- **For OpenAI SDK Integration:**
  - [OpenAI SDK Getting Started](./openai_sdk_agent_app_sample_code/README.md)
  - Navigate to `openai_sdk_agent_app_sample_code`
  - Examples of building agents using OpenAI SDK with MLflow PyFunc Models

## Repository Structure

```
├── agent_app_sample_code/             # Sample code for agent applications
│   ├── agents/                        # Agent code
│   ├── 03_agent_proof_of_concept.py   # Example of a proof of concept agent
│   └── ...                            # Additional directories and files
├── openai_sdk_agent_app_sample_code/  # Sample code using OpenAI SDK
│   └── ...                            # Directories and files
├── rag_app_sample_code/               # Sample code for RAG applications
│   ├── A_POC_app/                     # Proof-of-Concept applications
│   ├── pdf_uc_volume/                 # Example of a RAG application using a PDFs
│   ├── B_quality_iteration/           # Code for quality iteration
│   └── ...                            # Additional directories and files
├── genai_cookbook/                    # Documentation and learning materials
├── data/                              # Sample data for testing and development
├── dev/                               # Development tools and scripts
└── README.md                          # This README file
```

The `agent_app_sample_code` directory contains sample code for building agent applications using the Databricks platform.

The `openai_sdk_agent_app_sample_code` directory contains sample code that uses the OpenAI SDK + MLFlow PyFunc Models for building agents.

The `rag_app_sample_code` directory contains sample code for Retrieval-Augmented Generation (RAG) applications.

The `genai_cookbook` directory contains a 10 minute getting started guide.

The provided code is intended for use with the Databricks platform.  Specifically:
- [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) which provides a fast developer workflow with enterprise-ready LLMops & governance
- [Mosaic AI Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) which provides reliable, quality measurement using proprietary AI-assisted LLM judges to measure quality metrics that are powered by human feedback collected through an intuitive web-based chat UI

![Alt text](rag_app_sample_code/dbxquality.png)

## Getting Started

### Option 1: Running in a Databricks Workspace (Recommended)

1. **Clone the Repository into your Databricks Workspace:**
  - In your Databricks workspace, go to Repos
  - Click "Add Repo"
  - Enter the Git repository URL: `https://github.com/databricks/genai-cookbook.git`

2. **Set Up Your Databricks Environment:**
  - Use Serverless Notebooks or create a new cluster with Databricks Runtime 14.0 or higher

3. **Run the Sample Code:**
  - Navigate to `agent_app_sample_code/A_POC_app` to start with a proof of concept
  - Follow the numbered notebooks in sequence
  - Each notebook contains detailed instructions and explanations

### Option 2: Running Locally

1. **Prerequisites:**
  - Python 3.10 or higher
  - [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed and configured
  - Git installed on your local machine

2. **Clone and Set Up:**
  ```bash
  # Clone the repository
  git clone https://github.com/databricks/genai-cookbook.git
  cd genai-cookbook

  # Create and activate virtual environment
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate

  # Install dependencies
  pip install -r dev/dev_requirements.txt
  ```

3. **Configure Databricks Connection:**
  - Set up your Databricks CLI credentials:
    ```bash
    databricks configure --token
    ```
  - Follow the prompts to enter your Databricks workspace URL and access token

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

- For bugs or feature requests, [create an issue](https://github.com/databricks/genai-cookbook/issues)
- For questions, start a [GitHub Discussion](https://github.com/databricks/genai-cookbook/discussions)
