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


def get_table_url(table_fqdn):
    table_fqdn = table_fqdn.replace("`", "")
    catalog, schema, table = table_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/{catalog}/{schema}/{table}"
    return url


def get_volume_url(volume_fqdn):
    volume_fqdn = volume_fqdn.replace("`", "")
    catalog, schema, volume = volume_fqdn.split(".")
    browser_url = get_workspace_hostname()
    url = f"{browser_url}/explore/data/volumes/{catalog}/{schema}/{volume}"
    return url


def get_active_cluster_id() -> Optional[str]:
    """Get the active cluster ID.

    Returns:
        Optional[str]: The active cluster ID if found, None if not found or if an error occurs
    """
    if du.is_in_databricks_notebook():
        return du.get_active_cluster_id()
    else:
        return get_active_cluster_id_from_databricks_auth()


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
