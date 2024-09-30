from mlflow.utils import databricks_utils as du

# Helper function for display Delta Table URLs
def get_table_url(table_fqdn):
    table_fqdn = table_fqdn.replace("`", "")
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"https://{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url
