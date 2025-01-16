import mlflow
import json


def execute_function(tool, args):
    result = tool(**args)
    return json.dumps(result)
