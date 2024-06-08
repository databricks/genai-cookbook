# Iterating on quality
Once you have completed a baseline evaluation of your POC application, you will need to iteratively improve the application's quality.  Refer to [guidebook] for the steps to identify the root cause of quality issues and select which aspect of your RAG application's strategy you will experiment with changing.

There are 3 types of experiments you can run:
1. [Adjust the data pipeline](#1-adjust-the-data-pipeline)
    - *e.g., change the chunk sizes, parsing approach, etc*
    - *Note: To reflect the changed data pipeline, you will need to update the chain's configuration to point to the experiment's vector search*
2. [Adjust the chain's configuration](#2-adjust-the-chains-configuration)
    - *e.g., change the prompt, retrieval parameters, etc*
3. [Adjust the chain's code](#3-adjust-the-chains-code)
    - *e.g., add a re-ranker, guardrails, etc*

Across all types of experiments, the [01_evaluate_experiments](01_evaluate_experiments) is used to evaluate the experiments versus your baseline configuration (at first, this is your POC) and pick a "winner".  This notebook will help you pick the winning experiment and deploy it to the Review App or a production-ready, scalable REST API.

# Instructions for each approach
## 1. Adjust the data pipeline

### At a high level, you:
1. Configure and run a modified data pipeline to create a new Vector Index
2. Adjust your chain's configuration to use the the new Vector Index
3. Run evaluation on your updated chain to determine if your change improved quality/cost/latency

### Detailed steps
1. Decide which approach you want to use.
    1. **Run a single experiment at a time:** Allows you to configure and try a single data pipeline at once.  This mode is best if you want to try a single embedding model, test out a single new parser, etc.  We suggest starting here to get familar with these notebooks.
    2. **Run multiple experiments at once:** Also called a sweep, this approach allows you to, in parallel, execute multiple data pipelines at once.  This mode is best if you want to "sweep" across many different strategies, for example, evaluate 3 PDF parsers or evaluate many different chunk sizes.
2. Open the appropiate README and follow the steps there to create a new data pipeline and vector index.
    1. **Run a single experiment at a time:** [data_pipeline_experiments/single_experiment/README.md](./data_pipeline_experiments/single_experiment/README.md)
    2. **Run multiple experiments at once:** [data_pipeline_experiments/sweeps/README.md](./data_pipeline_experiments/sweeps/README.md)
3. After running the new data pipeline, the adjusted chain configuration will be stored inside an MLflow Run in your app's MLflow experiment.
4. Open the [01_evaluate_experiments](01_evaluate_experiments) Notebook and follow the steps there to run evaluation.


## 2. Adjust the chain's configuration

At a high level, you:
1. Adjust your chain's configuration
2. Run evaluation on your updated chain to determine if your change improved quality/cost/latency

### Detailed steps
1. Open the [01_evaluate_experiments](01_evaluate_experiments) Notebook and follow the steps there to create a modified chain configuration and run evaluation.

## 3. Adjust the chain's code

At a high level, you:
1. Modify your chain's code, optionally parameterizing the changes
2. Update the configuration to match the new parameters
3. Run evaluation on your updated chain to determine if your change improved quality/cost/latency

In this repo, we include several common chain code changes that can improve quality:
- **TODO** *Query rewriting:* Translating a user’s query into 1 or more queries that better represent the original intent in order to make the retrieval step more likely to find relevant documents. 
- **TODO** *Filter extraction:* Extracting specific filters from a user’s query that can be passed to the retrieval step.  This must be done in conjunction with changes to data pipeline to extract this metadata.
- *Re-ranking*: Retrieving a greater number of chunks, and then re-ranking them to identify a smaller number of most relevant chunks

### Detailed steps
1. Create a modified chain code file
2. Open the [01_evaluate_experiments](01_evaluate_experiments) Notebook and follow the steps there to evaluate the updated chain code.
