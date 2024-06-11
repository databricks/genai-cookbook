## **Step 1:** Clone code repo & create compute

```{image} ../images/5-hands-on/workflow_poc.png
:align: center
```
<br/>

The implement section is coupled with a repository of sample code designed to work on Databricks. 

Follow these steps to load the sample code to your Databricks workspace and configure the global settings for the application.

```{admonition} [Code Repository](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code)
:class: tip
You can find all of the sample code referenced throughout this section [here](https://github.com/databricks/genai-cookbook/tree/main/rag_app_sample_code).
```


### **Requirements**

1. A Databricks workspace with [serverless](https://docs.databricks.com/en/admin/workspace-settings/serverless.html) and Unity Catalog enabled
2. A [Mosaic AI Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) endpoint, either:
    - An existing endpoint
    - Permissions to create a new endpoint - the setup Notebook will do this for you
3. Unity Catalog Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored, either:
    - Write access to an existing [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) and [Schema](https://docs.databricks.com/en/data-governance/unity-catalog/index.html#the-unity-catalog-object-model) 
    - Permissions to create a new Unity Catalog and Schema - the setup Notebook will do this for you
4. A [**single-user**](https://docs.databricks.com/en/compute/configure.html#access-modes) cluster running [DBR 14.3+](https://docs.databricks.com/en/release-notes/runtime/index.html) with access to the internet
    - These tutorials have Python package conflicts with Machine Learning Runtime.  
    - Internet access is required to download the necessary Python and system packages 

### **Instructions**

1. Clone this repository into your workspace using [Git Folders](https://docs.databricks.com/en/repos/repos-setup.html)


    ```{image} ../images/5-hands-on/clone_repo.gif
    :align: center
    ```
<br/>

2. Open the [`rag_app_sample_code/00_global_config`](https://github.com/databricks/genai-cookbook/blob/main/rag_app_sample_code/00_global_config.py) Notebook and adjust the settings there.

    ```python
    # The name of the RAG application.  This is used to name the chain's UC model and prepended to the output Delta Tables + Vector Indexes
    RAG_APP_NAME = 'my_agent_app'

    # UC Catalog & Schema where outputs tables/indexs are saved
    # If this catalog/schema does not exist, you need create catalog/schema permissions.
    UC_CATALOG = f'{user_name}_catalog'
    UC_SCHEMA = f'rag_{user_name}'

    ## UC Model name where the POC chain is logged
    UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{RAG_APP_NAME}"

    # Vector Search endpoint where index is loaded
    # If this does not exist, it will be created
    VECTOR_SEARCH_ENDPOINT = f'{user_name}_vector_search'

    # Source location for documents
    # You need to create this location and add files
    SOURCE_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/source_docs"
    ```

3. Open and run the [`01_validate_config_and_create_resources`](https://github.com/databricks/genai-cookbook/blob/main/rag_app_sample_code/01_validate_config_and_create_resources.py) Notebook

> Proceed to the [Deploy POC](./5-hands-on-build-poc.md) step.
