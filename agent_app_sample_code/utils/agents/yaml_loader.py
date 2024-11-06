import os
from typing import List


def load_first_yaml_file(config_paths: List[str]) -> str:
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as handle:
                return handle.read()
    raise ValueError(
        f"No config file found at any of the following paths: {config_paths}. "
        f"Please ensure a config file exists at one of those paths."
    )
