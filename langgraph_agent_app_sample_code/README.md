# How to use local IDE

- databricks auth profile DEFAULT is set up
```
databricks auth profile login
```
- add a cluster_id in ~/.databrickscfg (if you want to use Spark code)
- add `openai_sdk_agent_app_sample_code/.env` to point to mlflow exp + dbx tracking uri (if you want to run any agent code from the terminal and have it logged to mlflow).  Make sure this mlflow experiment maps to the one in 02_agent_setup.ipynb.
```
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_NAME=/Users/your.name@company.com/my_agent_mlflow_experiment
```
- install poetry env & activate in your IDE
```
poetry install
```

if you want to use the data pipeline code in spark, you need to build the cookbook wheel and install it in the cluster
- build cookbook wheel
```
poetry build
```
- install cookbook wheel in cluster
    - Copy the wheel file to a UC Volume or Workspace folder
    - Go to the cluster's Libraries page and install the wheel file as a new library


