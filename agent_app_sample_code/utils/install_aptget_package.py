# Databricks notebook source
# MAGIC %md
# MAGIC #### `install_apt_get_packages`
# MAGIC
# MAGIC `install_apt_get_packages` installs specified apt-get packages required by a parser, ensuring that all necessary dependencies are available on the system.
# MAGIC
# MAGIC Arguments:
# MAGIC - `package_list (List[str])`: A list of apt-get packages to be installed.
# MAGIC
# MAGIC This function first cleans the apt-get cache, updates the package lists, and then installs the specified packages. It is designed to run across all nodes in a Spark cluster, using `mapPartitions` to execute the installation commands in parallel on each worker node.

# COMMAND ----------

from typing import List


def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser.

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(
        1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))
    )

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e
