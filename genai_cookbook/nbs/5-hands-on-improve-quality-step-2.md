### **Step 6:** Iteratively implement & evaluate quality fixes

```{image} ../images/5-hands-on/workflow_iterate.png
:align: center
```
<br/>

#### Requirements

1. Based on your [root cause analysis](./5-hands-on-improve-quality-step-1.md), you have identified a potential fixes to either [retrieval](./5-hands-on-improve-quality-step-1-retrieval.md) or [generation](./5-hands-on-improve-quality-step-1-generation.md) to implement and evaluate
2. Your POC application (or another baseline chain) is logged to an MLflow Run w/ a Quality Lab evaluation stored in the same Run

<!--
#### Overview

When working to improve the quality of the RAG system, changes can be broadly categorized into three buckets:

1. **![Data pipeline](../images/5-hands-on/data_pipeline.png)** Data pipeline changes
2. **![Chain config](../images/5-hands-on/chain_config.png)** RAG chain configuration changes
3. **![Chain code](../images/5-hands-on/chain_code.png)** RAG chain code changes

Depending on the specific issue you are trying to address, you may need to apply changes to one or both of these components. In some cases, simultaneous changes to both the data pipeline and RAG chain may be necessary to achieve the desired quality improvements.


##### Data pipeline changes

**Data pipeline changes** involve modifying how input data is processed, transformed, or stored before being used by the RAG chain. Examples of data pipeline changes include (and are not limited to):

- Trying a different chunking strategy
- Iterating on the document parsing process
- Changing the embedding model

Implementing a data pipeline change will generally require re-running the entire pipeline to create a new vector index. This process involves reprocessing the input documents, regenerating the vector embeddings, and updating the vector index with new embeddings and metadata.

##### RAG chain & code changes

**RAG chain changes** involve modifying steps or parameters of the RAG chain itself, without necessarily changing the underlying vector database. Examples of RAG chain changes include (and are not limited to):

- Changing the LLM
- Modifying the prompt template
- Adjusting the retrieval component (e.g., number of retrieval chunks, reranking, query expansion)
- Introducing additional processing steps such as a query understanding step

RAG chain updates may involve editing the **RAG chain configuration file** (e.g., changing the LLM parameters or prompt template), *or* modifying the actual **RAG chain code** (e.g., adding new processing steps or retrieval logic).


As a reminder, there are 3 types of potential fixes:
1. **![Data pipeline](../images/5-hands-on/data_pipeline.png)** 
    - *e.g., change the chunk sizes, parsing approach, etc*
    - *Note: To reflect the changed data pipeline, you will need to update the chain's configuration to point to the experiment's vector search*
2. **![Chain config](../images/5-hands-on/chain_config.png)**
    - *e.g., change the prompt, retrieval parameters, etc*
3. **![Chain code](../images/5-hands-on/chain_code.png)**
    - *e.g., add a re-ranker, guardrails, etc*
-->

#### Instructions
For all types, you will use the `B_quality_iteration/02_evaluate_fixes` Notebook to evaluate the resulting chain versus your baseline configuration (at first, this is your POC) and pick a "winner".  This notebook will help you pick the winning experiment and deploy it to the Review App or a production-ready, scalable REST API.

1. Open the `B_quality_iteration/02_evaluate_fixes` Notebook
2. Based on the type of fix you are implementing:
      - **![Data pipeline](../images/5-hands-on/data_pipeline.png)**
         1. Follow these [instructions](./5-hands-on-improve-quality-step-2-data-pipeline.md) to create the new data pipeline & get the name of the resulting MLflow Run.
         2. Add the run name(s) to the `DATA_PIPELINE_FIXES_RUN_NAMES` variable
      - **![Chain config](../images/5-hands-on/chain_config.png)** 
         1. Follow the instructions in the `Chain configuration` section of the `02_evaluate_fixes` Notebook to add chain configuration fixes to the `CHAIN_CONFIG_FIXES` variable.
      - **![Chain code](../images/5-hands-on/chain_code.png)**
         1. Create a modified chain code file and save it to the `B_quality_iteration/chain_code_fixes` folder.  Alternatively, select one of the provided chain code fixes from that folder.
         2. Follow the instructions in the `Chain code` section of the `02_evaluate_fixes` Notebook to add the chain code file and any additional chain configuration that is required to the `CHAIN_CODE_FIXES` variable
3. Run the notebook from the `Run evaluation` cell to
      - Evaluate each fix
      - Determine the fix with the best quality/cost/latency metrics
      - Deploy the best one to the Review App and a production-ready REST API to get stakeholder feedback
