# Helper functions for displaying Delta Table and Volume URLs

from typing import Optional
import json
import subprocess

from mlflow.utils import databricks_utils as du


def get_databricks_cli_config() -> dict:
    """Retrieve the Databricks CLI configuration by running 'databricks auth describe' command.

    Returns:
        dict: The parsed JSON configuration from the Databricks CLI, or None if an error occurs

    Note:
        Requires the Databricks CLI to be installed and configured
    """
    try:
        # Run databricks auth describe command and capture output
        process = subprocess.run(
            ["databricks", "auth", "describe", "-o", "json"],
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError if command fails
        )

        # Parse JSON output
        return json.loads(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running databricks CLI command: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing databricks CLI JSON output: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting databricks config from CLI: {e}")
        return None


def get_workspace_hostname() -> str:
    """Get the Databricks workspace hostname.

    Returns:
        str: The full workspace hostname (e.g., 'https://my-workspace.cloud.databricks.com')

    Raises:
        RuntimeError: If not in a Databricks notebook and unable to get workspace hostname from CLI config
    """
    if du.is_in_databricks_notebook():
        return "https://" + du.get_browser_hostname()
    else:
        cli_config = get_databricks_cli_config()
        if cli_config is None:
            raise RuntimeError("Could not get Databricks CLI config")
        try:
            return cli_config["details"]["host"]
        except KeyError:
            raise RuntimeError(
                "Could not find workspace hostname in Databricks CLI config"
            )


def get_table_url(table_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog table in the Databricks UI.

    Args:
        table_fqdn: Fully qualified table name in format 'catalog.schema.table'.
                   Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the table in the Databricks UI.

    Example:
        >>> get_table_url("main.default.my_table")
        'https://my-workspace.cloud.databricks.com/explore/data/main/default/my_table'
    """
    table_fqdn = table_fqdn.replace("`", "")
    catalog, schema, table = table_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/{catalog}/{schema}/{table}"
    return url


def get_volume_url(volume_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog volume in the Databricks UI.

    Args:
        volume_fqdn: Fully qualified volume name in format 'catalog.schema.volume'.
                    Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the volume in the Databricks UI.

    Example:
        >>> get_volume_url("main.default.my_volume")
        'https://my-workspace.cloud.databricks.com/explore/data/volumes/main/default/my_volume'
    """
    volume_fqdn = volume_fqdn.replace("`", "")
    catalog, schema, volume = volume_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/volumes/{catalog}/{schema}/{volume}"
    return url


def get_mlflow_experiment_url(experiment_id: str) -> str:
    """Generate the URL for an MLflow experiment in the Databricks UI.

    Args:
        experiment_id: The ID of the MLflow experiment

    Returns:
        str: The full URL to view the MLflow experiment in the Databricks UI.

    Example:
        >>> get_mlflow_experiment_url("<experiment_id>")
        'https://my-workspace.cloud.databricks.com/ml/experiments/<experiment_id>'
    """
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/ml/experiments/{experiment_id}"
    return url


def get_mlflow_experiment_traces_url(experiment_id: str) -> str:
    """Generate the URL for the MLflow experiment traces in the Databricks UI."""
    return get_mlflow_experiment_url(experiment_id) + "?compareRunsMode=TRACES"


def get_function_url(function_fqdn: str) -> str:
    """Generate the URL for a Unity Catalog function in the Databricks UI.

    Args:
        function_fqdn: Fully qualified function name in format 'catalog.schema.function'.
                      Can optionally include backticks around identifiers.

    Returns:
        str: The full URL to view the function in the Databricks UI.

    Example:
        >>> get_function_url("main.default.my_function")
        'https://my-workspace.cloud.databricks.com/explore/data/functions/main/default/my_function'
    """
    function_fqdn = function_fqdn.replace("`", "")
    catalog, schema, function = function_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/functions/{catalog}/{schema}/{function}"
    return url


def get_cluster_url(cluster_id: str) -> str:
    """Generate the URL for a Databricks cluster in the Databricks UI.

    Args:
        cluster_id: The ID of the cluster

    Returns:
        str: The full URL to view the cluster in the Databricks UI.

    Example:
        >>> get_cluster_url("<cluster_id>")
        'https://my-workspace.cloud.databricks.com/compute/clusters/<cluster_id>'
    """
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/compute/clusters/{cluster_id}"
    return url


def get_active_cluster_id_from_databricks_auth() -> Optional[str]:
    """Get the active cluster ID from the Databricks CLI authentication configuration.

    Returns:
        Optional[str]: The active cluster ID if found, None if not found or if an error occurs

    Note:
        This function relies on the Databricks CLI configuration having a cluster_id set
    """
    if du.is_in_databricks_notebook():
        raise ValueError(
            "Cannot get active cluster ID from the Databricks CLI in a Databricks notebook"
        )
    try:
        # Get config from the databricks cli
        auth_output = get_databricks_cli_config()

        # Safely navigate nested dict
        details = auth_output.get("details", {})
        config = details.get("configuration", {})
        cluster = config.get("cluster_id", {})
        cluster_id = cluster.get("value")

        if cluster_id is None:
            raise ValueError("Could not find cluster_id in Databricks auth config")

        return cluster_id

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def get_active_cluster_id() -> Optional[str]:
    """Get the active cluster ID.

    Returns:
        Optional[str]: The active cluster ID if found, None if not found or if an error occurs
    """
    if du.is_in_databricks_notebook():
        return du.get_active_cluster_id()
    else:
        return get_active_cluster_id_from_databricks_auth()
