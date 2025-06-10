import mlflow
import json


@mlflow.trace(span_type="FUNCTION")
def execute_function(tool, args):
    result = tool(**args)
    return json.dumps(result)
