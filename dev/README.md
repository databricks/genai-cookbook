# Databricks Mosaic Generative AI Cookbook

## Dev env setup
- clone the repo; `cd cookbook`
- use your preferred approach to starting a new python environment
- in that environment, `pip install -r dev/dev_requirements.txt`

## Updating website content
To test updates to site content at ai-cookbook.io
- build and preview the site with `jupyter-book build --all genai_cookbook`

The homepage is at `genai_cookbook/index.md`

The content pages are in `genai_cookbook/nbs/`

Jupyter book is fairly flexible and offers a lot of different options for formatting, cross-referencing, adding formatted callouts, etc. Read more at the [Jupyter Book docs](https://jupyterbook.org/en/stable/intro.html).

## Updating code
Use the `databricks sync` CLI command ([docs](https://docs.databricks.com/en/dev-tools/cli/sync-commands.html)) to sync the code in this repo to 
your Databricks workspace. You can then iterate on code in your IDE and test changes in 
Databricks. Be sure to add unit tests (as of the time of writing, tests are under `agent_app_sample_code/tests`).
You can run unit tests via `pytest`
